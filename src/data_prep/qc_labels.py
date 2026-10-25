"""
标注质检（比 check_dataset.py 更细，训练前跑一遍）
==================================================
检查项：
  1. 图片 / 标注配对完整性
  2. 文件名前缀（类别名）与标注类别 id 是否一致 —— 能直接抓出"框对了但类别选错"
  3. 同类重复框（IoU 高 或 高度包含）—— 单实例地标应该一图一类一框
  4. 框面积分布（极小框多为噪点/误标）
  5. 坐标越界、宽高为 0
  6. dataset/ 与 labeled/ 的一致性（split 是否最新）

用法：
  python src/qc_labels.py                 # 检查 data/labeled（源标注）
  python src/qc_labels.py --iou 0.5       # 自定义重复框阈值
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_classes(path="data/classes.txt"):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        n = line.split("#", 1)[0].strip()
        if n:
            names.append(n)
    return names


def to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return (inter / union if union > 0 else 0.0), inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/labeled")
    ap.add_argument("--classes", default="data/classes.txt")
    ap.add_argument("--iou", type=float, default=0.3, help="判为重框的 IoU 阈值")
    ap.add_argument("--contain", type=float, default=0.7, help="判为重框的包含度阈值")
    ap.add_argument("--tiny", type=float, default=0.002, help="极小框面积阈值（占全图比例）")
    args = ap.parse_args()

    names = load_classes(args.classes)
    nc = len(names)
    src = Path(args.src)
    imgs = {p.stem: p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS}

    stats = defaultdict(lambda: {"imgs": 0, "boxes": 0})
    per_img_boxcnt = Counter()
    mismatched, dup, tiny, oob, bad_fmt, empty, no_lbl = [], [], [], [], [], [], []

    for stem, img in sorted(imgs.items()):
        lbl = src / "labels" / (stem + ".txt")
        if not lbl.exists():
            no_lbl.append(stem)
            continue
        raw = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not raw:
            empty.append(stem)
            per_img_boxcnt[0] += 1
            continue
        per_img_boxcnt[len(raw)] += 1
        boxes = []
        for line in raw:
            parts = line.split()
            if len(parts) != 5:
                bad_fmt.append(f"{stem}: '{line}'")
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                bad_fmt.append(f"{stem}: '{line}'")
                continue
            if not (0 <= cid < nc):
                oob.append(f"{stem}: cid={cid}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                oob.append(f"{stem}: '{line}'")
                continue
            boxes.append((cid, cx, cy, bw, bh))
            stats[names[cid]]["boxes"] += 1

        # 文件名前缀 == 标注类别？（文件名形如 sydney_opera_000059）
        if boxes:
            prefixes = [n for n in names if stem.startswith(n)]
            prefix = max(prefixes, key=len) if prefixes else None
            got = {names[c] for c, *_ in boxes}
            if prefix and got != {prefix}:
                mismatched.append(f"{stem} (文件名={prefix}, 标注={sorted(got)})")

        # 同类重复框
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] != boxes[j][0]:
                    continue
                ra, rb = to_xyxy(*boxes[i][1:]), to_xyxy(*boxes[j][1:])
                iou_v, inter = overlap(ra, rb)
                smaller = min((ra[2] - ra[0]) * (ra[3] - ra[1]), (rb[2] - rb[0]) * (rb[3] - rb[1]))
                ct = inter / smaller if smaller > 0 else 0.0
                if iou_v > args.iou or ct > args.contain:
                    dup.append(f"{stem}: {names[boxes[i][0]]} IoU={iou_v:.2f} contain={ct:.2f}")

        for c, cx, cy, bw, bh in boxes:
            if bw * bh < args.tiny:
                tiny.append(f"{stem}: {names[c]} area={bw * bh * 100:.2f}%")

    for n in names:
        stats[n]["imgs"] = sum(1 for s in imgs if s.startswith(n))

    print(f"===== 标注质检（{len(imgs)} 张图片 / {nc} 类）=====")
    print(f"{'类别':<22}{'图片数':>6}{'框数':>6}")
    for n in names:
        print(f"{n:<22}{stats[n]['imgs']:>6}{stats[n]['boxes']:>6}")
    print(f"{'合计':<20}{len(imgs):>6}{sum(s['boxes'] for s in stats.values()):>6}")
    print(f"\n每张图目标数分布: {dict(sorted(per_img_boxcnt.items()))}")

    def report(title, items, limit=25):
        if items:
            print(f"\n[!] {title}: {len(items)}")
            for it in items[:limit]:
                print("   -", it)
            if len(items) > limit:
                print(f"   ... 其余 {len(items) - limit} 条省略")

    report("缺少标注文件", no_lbl)
    report("空标注（负样本）", empty)
    report("格式错误", bad_fmt)
    report("类别 id / 坐标越界", oob)
    report("文件名类别与标注类别不一致", mismatched)
    report("同类重复框（一图多框，需合并）", dup)
    report("极小框（疑似噪点）", tiny)

    # dataset 一致性
    ds = Path("data/dataset")
    if (ds / "images").exists():
        ds_stems, miss_pair = set(), []
        for sp in ("train", "val", "test"):
            d = ds / "images" / sp
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                ds_stems.add(p.stem)
                if not (ds / "labels" / sp / (p.stem + ".txt")).exists():
                    miss_pair.append(p.name)
        print(f"\n===== dataset 一致性 =====")
        print(f"dataset 图片 {len(ds_stems)} / labeled 图片 {len(imgs)}")
        print(f"dataset 多出（上次 split 残留，重跑 split 前需清空）: {len(ds_stems - set(imgs))}")
        print(f"dataset 缺失（新标注未纳入划分）: {len(set(imgs) - ds_stems)}")
        if miss_pair:
            print(f"dataset 中缺 label 的图片: {len(miss_pair)} 张，例如 {miss_pair[:5]}")

    ok = not (no_lbl or bad_fmt or oob or mismatched)
    print("\n结论:", "结构性检查通过" if ok else "存在必须修复的问题（见上）")
    print("提示: 若'同类重复框'非空，先跑 python src/merge_boxes.py 再 split")


if __name__ == "__main__":
    main()
