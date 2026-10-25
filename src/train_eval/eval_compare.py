"""
多模型对比评估（在 test 集上跑 mAP，输出 Markdown 表格）
======================================================
用法：
  python src/eval_compare.py --weights models/best.pt runs/detect/xxx/weights/best.pt
  python src/eval_compare.py --weights models/best.pt --split test --imgsz 320

输出：逐模型整体指标 + 每类 mAP 对比表，可直接贴进报告。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    names = YOLO(args.weights[0]).names
    rows = {}
    for w in args.weights:
        if not Path(w).exists():
            print(f"[跳过] 不存在: {w}")
            continue
        print(f"\n>>> 评估 {w}")
        m = YOLO(w)
        r = m.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  batch=args.batch, device="cpu", plots=False, verbose=False)
        label = Path(w).parent.parent.name if Path(w).name == "best.pt" else w
        entry = {"all": {"P": r.box.mp, "R": r.box.mr, "mAP50": r.box.map50, "mAP50-95": r.box.map}}
        for i, cname in enumerate(names):
            if i < len(r.box.ap50):
                entry[cname] = {"P": float(r.box.p[i]) if i < len(r.box.p) else float("nan"),
                                "R": float(r.box.r[i]) if i < len(r.box.r) else float("nan"),
                                "mAP50": float(r.box.ap50[i]),
                                "mAP50-95": float(r.box.ap[i])}
        rows[label] = entry

    if not rows:
        raise SystemExit("没有可评估的模型")

    print(f"\n\n===== {args.split} 集对比（imgsz={args.imgsz}）=====")
    hdr = "| 类别 | " + " | ".join(f"{k} mAP50 / mAP50-95" for k in rows) + " |"
    print(hdr)
    print("|" + "---|" * (len(rows) + 1))
    for key in ["all", *names]:
        if key not in next(iter(rows.values())):
            continue
        cells = []
        for k, v in rows.items():
            c = v.get(key)
            cells.append(f"{c['mAP50']:.3f} / {c['mAP50-95']:.3f}" if c else "-")
        label = "**全部**" if key == "all" else key
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
