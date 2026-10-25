"""
图像推理脚本（演示"能够识别图像"）
=================================
用法：
  python src/predict_image.py --source data/demo.jpg
  python src/predict_image.py --source data/test_imgs/            # 整个文件夹
  python src/predict_image.py --source data/demo.jpg --save       # 保存画框结果图
  python src/predict_image.py --source data/demo.jpg --openvino   # 用 OpenVINO 加速模型
权重查找顺序：models/best.pt -> runs/detect/*/weights/best.pt（按时间取最新）
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def find_weights(use_openvino: bool) -> str:
    if use_openvino:
        ov_dir = Path("models/best_openvino_model")
        if ov_dir.exists():
            return str(ov_dir)
        print("[警告] 未找到 models/best_openvino_model，回退 best.pt（先运行 src/export_openvino.py）")
    if Path("models/best.pt").exists():
        return "models/best.pt"
    # 递归查找：ultralytics 会把相对 project 拼到 runs_dir 下，实际产物可能出现在
    # runs/detect/<name>/ 或 runs/detect/runs/detect/<name>/，用 rglob 两种都能命中
    cands = sorted(Path("runs").rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if cands:
        print(f"[提示] 使用最近一次训练的权重: {cands[-1]}")
        return str(cands[-1])
    raise SystemExit("找不到权重文件：请先训练（src/train.py），或把 best.pt 放到 models/ 下")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片路径或文件夹")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.4, help="NMS IoU 阈值，越小抑制越强")
    ap.add_argument("--max-det", type=int, default=0, help="每图最多保留几个目标；0=不限，演示可用 1")
    ap.add_argument("--save", action="store_true", help="保存画框结果图到 runs/detect/predict/")
    ap.add_argument("--openvino", action="store_true")
    args = ap.parse_args()

    weights = args.weights or find_weights(args.openvino)
    print(f"加载模型: {weights}")
    model = YOLO(weights)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det if args.max_det > 0 else 300,
        device="cpu",
        save=args.save,
        project=str(Path("runs/detect").resolve()),
        name="predict",
        exist_ok=True,
        verbose=False,
    )
    for r in results:
        fname = Path(getattr(r, "path", "?")).name
        if r.boxes is None or len(r.boxes) == 0:
            print(f"{fname}: 未检测到目标")
            continue
        for b in r.boxes:
            cls = r.names[int(b.cls)]
            conf = float(b.conf)
            xyxy = [round(v, 1) for v in b.xyxy[0].tolist()]
            print(f"{fname}: {cls} (conf={conf:.2f}) box={xyxy}")


if __name__ == "__main__":
    main()
