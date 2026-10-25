"""
数据集质检：训练前必跑，能挡掉 80% 的"训练完发现效果差"问题
=========================================================
检查项：
  1. 图片能否正常打开、尺寸是否过小
  2. 每张图片是否有对应标注文件
  3. 类别 id 是否在 [0, nc) 范围内
  4. 归一化坐标是否越界（YOLO 要求 cx,cy,w,h 都在 0~1）
  5. 空标注文件（可作负样本保留，但会提示数量）
  6. 统计每类目标数、每张图目标数分布
"""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--nc", type=int, required=True, help="类别数，与 data.yaml 的 nc 一致")
    args = ap.parse_args()

    root = Path(args.root)
    errors, warns = [], []
    cls_counter = Counter()
    per_img_obj = Counter()
    total_imgs = 0

    for split in ["train", "val", "test"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            warns.append(f"{img_dir} 不存在，跳过")
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if not imgs:
            warns.append(f"{img_dir} 下没有图片")
            continue
        total_imgs += len(imgs)
        for img in imgs:
            # 1) 图片可打开且尺寸不太小
            try:
                with Image.open(img) as im:
                    w, h = im.size
                if min(w, h) < 64:
                    warns.append(f"图片过小 {w}x{h}: {img}")
            except Exception as e:
                errors.append(f"图片损坏: {img} ({e})")
                continue
            # 2) 标注存在性
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                errors.append(f"缺少标注: {img}")
                continue
            # 3) 4) 标注内容
            lines = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                warns.append(f"空标注（负样本）: {img}")
                per_img_obj[0] += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"标注列数不为 5: {lbl} -> '{line}'")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    errors.append(f"标注数值解析失败: {lbl} -> '{line}'")
                    continue
                if not (0 <= cid < args.nc):
                    errors.append(f"类别 id 越界(0~{args.nc - 1}): {lbl} -> cid={cid}")
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    errors.append(f"坐标越界(0~1): {lbl} -> '{line}'")
                if bw <= 0 or bh <= 0:
                    errors.append(f"框宽/高为 0: {lbl} -> '{line}'")
                cls_counter[cid] += 1
            per_img_obj[len(lines)] += 1

    print(f"===== 质检结果（{total_imgs} 张图片）=====")
    print(f"每类目标数量（类别 id: 数量）: {dict(sorted(cls_counter.items()))}")
    print(f"每张图目标数分布: {dict(sorted(per_img_obj.items()))}")
    if warns:
        print(f"\n[警告] {len(warns)} 条（不阻塞训练，但建议处理）:")
        for w in warns[:15]:
            print("  -", w)
    if errors:
        print(f"\n[错误] {len(errors)} 条（必须修完再训练）:")
        for e in errors[:20]:
            print("  -", e)
        raise SystemExit(1)
    print("\n全部通过，可以开始训练。")


if __name__ == "__main__":
    main()
