"""
合并同类重叠框（标注清理，训练前必做）
=====================================
背景：AI 预标注常对同一目标产出多个重叠框（如悉尼歌剧院被框了 3~4 个），
      人工复核时容易漏掉。多个重叠 GT 框会让模型学会"一个目标输出多个框"，
      推理时出现重复检测，同时拉低 mAP。
      地标类属于单实例目标，正确约定是"一栋建筑一个框"。

规则：同一类别内，两框 IoU > 阈值 或 一个框被另一个基本包含（交集/较小框面积 > 0.7）
      则归为一簇，簇内取外接框（并集），不相交的框保持独立。

用法：
  python src/merge_boxes.py --dry-run     # 只统计不修改
  python src/merge_boxes.py               # 执行合并（覆盖 data/labeled/labels/*.txt）
"""
import argparse
from pathlib import Path

LBL_DIR = Path("data/labeled/labels")


def to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return (inter / union if union > 0 else 0.0), inter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.2, help="判定同簇的 IoU 阈值")
    ap.add_argument("--contain", type=float, default=0.7, help="包含度阈值（交集/较小框面积）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = 0
    before = after = 0
    for lbl in sorted(LBL_DIR.glob("*.txt")):
        lines = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        boxes = [(int(p[0]), *map(float, p[1:])) for p in lines if len(p) == 5]
        if not boxes:
            continue
        before += len(boxes)
        by_cls = {}
        for cid, cx, cy, w, h in boxes:
            by_cls.setdefault(cid, []).append(to_xyxy(cx, cy, w, h))
        merged = []
        for cid, rects in by_cls.items():
            n = len(rects)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(n):
                for j in range(i + 1, n):
                    i_ov, inter = iou(rects[i], rects[j])
                    smaller = min(
                        (rects[i][2] - rects[i][0]) * (rects[i][3] - rects[i][1]),
                        (rects[j][2] - rects[j][0]) * (rects[j][3] - rects[j][1]),
                    )
                    contain = inter / smaller if smaller > 0 else 0.0
                    if i_ov > args.iou or contain > args.contain:
                        parent[find(i)] = find(j)
            clusters = {}
            for i in range(n):
                clusters.setdefault(find(i), []).append(rects[i])
            for rects_c in clusters.values():
                x1 = min(r[0] for r in rects_c)
                y1 = min(r[1] for r in rects_c)
                x2 = max(r[2] for r in rects_c)
                y2 = max(r[3] for r in rects_c)
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(1.0, x2), min(1.0, y2)
                merged.append((cid, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        after += len(merged)
        if len(merged) != len(boxes):
            changed += 1
            if not args.dry_run:
                lbl.write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in merged),
                    encoding="utf-8",
                )
    mode = "统计" if args.dry_run else "已合并"
    print(f"[{mode}] 受影响文件 {changed} 个；框数 {before} -> {after}")
    if args.dry_run:
        print("确认无误后去掉 --dry-run 执行")


if __name__ == "__main__":
    main()
