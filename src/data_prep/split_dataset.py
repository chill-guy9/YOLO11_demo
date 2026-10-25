"""
数据集划分：把标注好的数据按 8:1:1 切成 train/val/test，并重建 data.yaml
====================================================================
预期目录结构（平坦式；标注完成后，把图片和 txt 分别拷进来）：
  data/labeled/images/*.jpg        # 所有图片
  data/labeled/labels/*.txt        # 与图片同名的 YOLO 标注
类别定义在 data/classes.txt：每行一个类别名，行号（从 0 开始）即类别 id。

用法：
  python src/split_dataset.py                     # 默认 8:1:1，seed=42
  python src/split_dataset.py --ratio 0.8 0.1 0.1 # 显式指定比例
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/labeled")
    ap.add_argument("--dst", default="data/dataset")
    ap.add_argument("--classes", default="data/classes.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    names = []
    for line in Path(args.classes).read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()  # 支持行内中文注释
        if name:
            names.append(name)
    if not names:
        raise SystemExit(f"{args.classes} 里没有有效类别名")

    pairs, missing = [], []
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = src / "labels" / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing.append(img.name)
    if missing:
        print(f"[警告] {len(missing)} 张图片缺少标注文件，未纳入数据集，例如: {missing[:5]}")
    if not pairs:
        raise SystemExit(f"{src}/images 下没有找到(图片+标注)配对，请先完成标注")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_tr = int(n * args.ratio[0])
    n_va = int(n * args.ratio[1])
    splits = {
        "train": pairs[:n_tr],
        "val": pairs[n_tr : n_tr + n_va],
        "test": pairs[n_tr + n_va :],
    }

    for split, items in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, dst / "images" / split / img.name)
            shutil.copy2(lbl, dst / "labels" / split / lbl.name)
        print(f"{split:5s}: {len(items):4d} 张")

    yaml_text = (
        f"# 由 src/split_dataset.py 自动生成，请勿手改（改了也会被重新覆盖）\n"
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    Path("data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"总计 {n} 对 (图片+标注)，已生成 data.yaml，类别 {len(names)} 个: {names}")


if __name__ == "__main__":
    main()
