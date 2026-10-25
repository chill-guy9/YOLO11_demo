"""
定向补数据：爬取 -> 去重 -> YOLO-World 预标注 -> 严格筛选 -> 暂存待抽检
=====================================================================
用途：只补特定类别（当前是 great_wall / oriental_pearl），不重跑全量采集。
关键设计：
  1. **不直接写入 data/labeled/**：先落到 data/_supp_staging/，人工抽检通过后再并入，
     避免自动标注的噪声进入训练集。
  2. 去重同时比 md5 和 pHash（汉明距离 <= 6 视为同图），跟已有 raw/labeled 图片也去重。
  3. 只保留"本类"的框，取面积最大的一个（地标是单实例），并按类设面积下限，
     过滤掉框住局部砖石/招牌的误检。
  4. 输出预览网格 data/_supp_staging/<cls>_preview.jpg，供肉眼确认框质量。

用法：
  python src/supplement.py --classes great_wall oriental_pearl --target 150
  python src/supplement.py --classes great_wall --target 150 --dry-run   # 只看爬取与去重结果
"""
import argparse
import hashlib
import shutil
import time
from pathlib import Path

import imagehash
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 每类的检索词（多角度/多季节，提高类内多样性）
KEYWORDS = {
    "great_wall": [
        "Great Wall of China Mutianyu",
        "慕田峪长城",
        "司马台长城",
        "金山岭长城 秋色",
        "Great Wall of China Jinshanling",
    ],
    "oriental_pearl": [
        "Oriental Pearl Tower Shanghai",
        "东方明珠 上海 陆家嘴",
        "东方明珠塔 夜景",
        "Oriental Pearl Tower night skyline",
        "Shanghai Lujiazui skyline",
    ],
}
# YOLO-World 文本提示（每类只喂一个提示，检测更聚焦）
PROMPTS = {
    "great_wall": "Great Wall of China",
    "oriental_pearl": "Oriental Pearl Tower",
}
# 框面积下限（归一化）：长城是大目标，塔是细高目标
MIN_AREA = {"great_wall": 0.10, "oriental_pearl": 0.03}
MIN_SIDE = 0.02  # 框宽高下限，过滤细条误检


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_of(path):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB").resize((256, 256)))
    except Exception:
        return None


def collect_existing(cls):
    """已有图片的 md5/phash 集合（raw + labeled 两处都算）"""
    md5s, phs = set(), []
    for d in (Path("data/data_raw") / cls, Path("data/labeled/images")):
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if d.name == "images" and not p.stem.startswith(cls):
                continue
            md5s.add(md5_of(p))
            h = phash_of(p)
            if h is not None:
                phs.append(h)
    return md5s, phs


def crawl(keywords, out_dir, per_query):
    from icrawler.builtin import BingImageCrawler

    out_dir.mkdir(parents=True, exist_ok=True)
    for kw in keywords:
        print(f"  [crawl] {kw}")
        try:
            BingImageCrawler(
                storage={"root_dir": str(out_dir)},
                parser_threads=2,
                downloader_threads=4,
            ).crawl(keyword=kw, max_num=per_query, file_idx_offset="auto")
        except Exception as e:
            print(f"  [warn] 关键词失败 {kw}: {type(e).__name__}: {e}")
    return sorted(p for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def make_preview(items, out_path, cell=300, cols=5):
    """items: [(img_path, (cx,cy,w,h), conf)] -> 预览网格，框已画出"""
    import cv2
    import numpy as np

    if not items:
        return
    n = min(len(items), 20)
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 240, dtype=np.uint8)
    for i, (p, box, conf) in enumerate(items[:n]):
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy, bw, bh = box
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 140, 255), 2)
        s = min(cell / w, cell / h)
        im = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - im.shape[0]) // 2
        x0 = c * cell + (cell - im.shape[1]) // 2
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_path))
    print(f"  [preview] {out_path}（前 {n} 张）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["great_wall", "oriental_pearl"])
    ap.add_argument("--target", type=int, default=150, help="每类目标总张数")
    ap.add_argument("--per-query", type=int, default=60, help="每组关键词爬取数量")
    ap.add_argument("--conf", type=float, default=0.30, help="预标注置信度下限")
    ap.add_argument("--dedup-dist", type=int, default=6, help="pHash 汉明距离阈值")
    ap.add_argument("--dry-run", action="store_true", help="只爬取+去重，不写标注")
    args = ap.parse_args()

    staging = Path("data/_supp_staging")
    staging.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO("yolov8s-worldv2.pt")

    summary = []
    for cls in args.classes:
        if cls not in KEYWORDS:
            print(f"[跳过] 未配置关键词: {cls}")
            continue
        have = len([p for p in (Path("data/data_raw") / cls).iterdir()
                    if p.suffix.lower() in IMG_EXTS]) if (Path("data/data_raw") / cls).exists() else 0
        need = max(0, args.target - have)
        print(f"\n===== {cls}：现有 {have} 张，目标 {args.target}，需补 {need} =====")
        if need <= 0:
            print("  已达标，跳过")
            continue

        cand_dir = staging / cls / f"cand_{time.strftime('%Y%m%d_%H%M%S')}"
        files = crawl(KEYWORDS[cls], cand_dir, args.per_query)
        print(f"  爬取候选 {len(files)} 张")

        md5s, phs = collect_existing(cls)
        uniq, seen_md5, seen_ph = [], set(), []
        dup_md5 = dup_ph = 0
        for p in files:
            m = md5_of(p)
            if m in md5s or m in seen_md5:
                dup_md5 += 1
                continue
            h = phash_of(p)
            if h is not None and (any(h - x <= args.dedup_dist for x in phs)
                                  or any(h - x <= args.dedup_dist for x in seen_ph)):
                dup_ph += 1
                continue
            seen_md5.add(m)
            if h is not None:
                seen_ph.append(h)
            uniq.append(p)
        print(f"  去重后 {len(uniq)} 张（md5 重复 {dup_md5}，pHash 近似 {dup_ph}）")

        if args.dry_run:
            summary.append((cls, len(files), len(uniq), 0))
            continue

        # 预标注 + 筛选
        model.set_classes([PROMPTS[cls]])
        min_area, min_side = MIN_AREA[cls], MIN_SIDE
        accepted, rejected = [], {"无框": 0, "面积过小": 0, "读图失败": 0}
        out_img = staging / cls / "images"
        out_lbl = staging / cls / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, p in enumerate(uniq, 1):
            try:
                r = model.predict(str(p), imgsz=640, conf=0.15, iou=0.5,
                                  device="cpu", verbose=False)[0]
            except Exception:
                rejected["读图失败"] += 1
                continue
            boxes = []
            if r.boxes is not None:
                for b in r.boxes:
                    cx, cy, w, h = b.xywhn[0].tolist()
                    conf = float(b.conf)
                    if conf < args.conf or w < min_side or h < min_side:
                        continue
                    if w * h < min_area:
                        continue
                    boxes.append((conf, (cx, cy, w, h)))
            if not boxes:
                rejected["无框" if r.boxes is None or len(r.boxes) == 0 else "面积过小"] += 1
                continue
            best_conf, best_box = max(boxes, key=lambda x: x[0] * x[1][2] * x[1][3])
            stem = f"{cls}_sup{i:04d}"
            data = p.read_bytes()
            (out_img / (stem + p.suffix.lower())).write_bytes(data)
            cx, cy, w, h = best_box
            (out_lbl / (stem + ".txt")).write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
                                                  encoding="utf-8")
            accepted.append((p, best_box, best_conf))
            if i % 20 == 0:
                print(f"    ...{i}/{len(uniq)}，已通过 {len(accepted)}")

        print(f"  筛选完成：通过 {len(accepted)} / {len(uniq)}，"
              f"淘汰 {dict((k, v) for k, v in rejected.items() if v)}，"
              f"耗时 {(time.time() - t0) / 60:.1f} 分钟")
        make_preview(accepted, staging / f"{cls}_preview.jpg")
        summary.append((cls, len(files), len(uniq), len(accepted)))
        # 候选原图保留（不再 rmtree）：WorkBuddy 的批量删除保护会拦截 >50 文件的删除，
        # 且保留原始候选便于事后复查/补标

    print("\n===== 补数据汇总 =====")
    print(f"{'类别':<20}{'爬取':>6}{'去重后':>8}{'通过筛选':>10}")
    for cls, n1, n2, n3 in summary:
        print(f"{cls:<18}{n1:>6}{n2:>8}{n3:>10}")
    print(f"\n暂存目录: {staging.resolve()}")
    print("下一步：看 <cls>_preview.jpg 抽检框质量 -> 通过后执行 src/merge_supplement.py 并入训练集")


if __name__ == "__main__":
    main()
