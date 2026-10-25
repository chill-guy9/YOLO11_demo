"""
AI 预标注（强烈推荐）：用 YOLO-World 开放词表检测给待标注图片打初版框
====================================================================
原理：YOLO-World 支持"文本提示"零样本检测——把 6 个类别名（英文提示词）喂给它，
      不需要先训练就能出框。GLDv2 已给出每张图的类别（来自所在文件夹），
      所以这里只做"已知类别下的框定位"：只保留该类别的框，精度比全图检测更稳。

输入：data/data_raw/<类别名>/*.jpg        （download_gldv2.py 下载的图片）
输出：data/labeled/images/*.jpg      （全部图片拷贝过来，便于统一标注）
      data/labeled/labels/*.txt      （YOLO 格式初版标注；未检出的图写空文件）
      data/labeled/_needs_manual.txt （未检出预期类别的图，需人工补框）

警告：预标注框必须经人工复核/修正后才能用于训练！
      用 X-AnyLabeling 打开 data/labeled/images/（文件->打开目录，导入 YOLO 格式）逐张检查。

依赖：首次运行自动下载 yolov8s-worldv2.pt（约 30MB+，需能访问 GitHub）
用法：python src/auto_label.py                  # 全量预标注
      python src/auto_label.py --limit 5        # 每类先试 5 张看效果
"""
import argparse
import shutil
import time
from pathlib import Path

from PIL import Image

RAW = Path("data/data_raw")
OUT_IMG = Path("data/labeled/images")
OUT_LBL = Path("data/labeled/labels")
CLASSES_FILE = Path("data/classes.txt")

# 类别名 -> YOLO-World 文本提示词。
# 按 data/classes.txt 的顺序自动组装成 PROMPTS，增删类别只需改这个映射，
# 不会再出现"classes.txt 有 5 类、脚本内置 6 个提示词"而对不上的情况。
PROMPT_MAP = {
    "eiffel_tower": "Eiffel Tower",
    "sydney_opera": "Sydney Opera House",
    "oriental_pearl": "Oriental Pearl Tower",
    "statue_of_liberty": "Statue of Liberty",
    "great_wall": "Great Wall of China",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_classes():
    names = []
    for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines():
        n = line.split("#", 1)[0].strip()
        if n:
            names.append(n)
    missing = [n for n in names if n not in PROMPT_MAP]
    if missing:
        raise SystemExit(
            f"以下类别在 classes.txt 里存在，但 PROMPT_MAP 缺少对应提示词: {missing}；"
            f"请在本脚本顶部补上，例如 \"{missing[0]}\": \"<英文提示词>\""
        )
    return names


def load_prompts(names):
    """按 classes.txt 的顺序返回提示词列表（列表下标即类别 id）"""
    return [PROMPT_MAP[n] for n in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15, help="预标注置信度阈值，宜低不宜高")
    ap.add_argument("--limit", type=int, default=0, help="每类最多处理几张（调试用，0=全部）")
    args = ap.parse_args()

    names = load_classes()
    prompts = load_prompts(names)
    print(f"类别（{len(names)}）: {list(zip(names, prompts))}")

    from ultralytics import YOLO

    model = YOLO(args.weights)  # 首次自动下载
    model.set_classes(prompts)  # 开放词表：提示词顺序即类别 id 顺序

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    total = detected = 0
    needs_manual = []
    t0 = time.time()
    for cls_id, cls_name in enumerate(names):
        src_dir = RAW / cls_name
        if not src_dir.exists():
            print(f"[跳过] {src_dir} 不存在")
            continue
        imgs = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.limit:
            imgs = imgs[: args.limit]
        n_hit = 0
        for img_path in imgs:
            total += 1
            # 文件名必须加类别前缀：icrawler 在每个类别目录下都从 000001.jpg 开始编号，
            # 不同类别的同名文件直接拷贝会互相覆盖（曾因此丢数据）
            dst_name = f"{cls_name}_{img_path.name}"
            dst_img = OUT_IMG / dst_name
            try:
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                results = model.predict(
                    source=str(img_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=0.5,
                    device="cpu",
                    verbose=False,
                )
            except Exception as e:
                print(f"  [跳过] {img_path.name}: {type(e).__name__}: {e}")
                if dst_img.exists():
                    dst_img.unlink()
                continue
            lines = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    if int(b.cls) != cls_id:  # 只保留本图应有类别的框
                        continue
                    cx, cy, w, h = b.xywhn[0].tolist()
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (OUT_LBL / f"{cls_name}_{img_path.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            if lines:
                detected += 1
                n_hit += 1
            else:
                needs_manual.append(f"{cls_name}/{img_path.name}")
        print(f"[{cls_name}] {len(imgs)} 张，检出 {n_hit} 张")

    report = Path("data/labeled/_needs_manual.txt")
    report.write_text("\n".join(needs_manual) + "\n", encoding="utf-8")
    elapsed = time.time() - t0
    print(
        f"\n预标注完成：{total} 张中检出 {detected} 张"
        f"（{detected / max(total, 1):.0%}），耗时 {elapsed / 60:.1f} 分钟"
    )
    print(f"未检出 {len(needs_manual)} 张（需人工补框），清单见 {report}")
    print("下一步：用 X-AnyLabeling 打开 data/labeled/images/ 复核修正所有框")


if __name__ == "__main__":
    main()
