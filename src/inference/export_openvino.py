"""
导出 OpenVINO 加速模型（CPU 推理通常比 PyTorch 快 2~3 倍）
===========================================================
用法：
  python src/export_openvino.py                    # FP32，无损加速
  python src/export_openvino.py --int8              # INT8 量化（需 pip install nncf，精度略降）
导出后推理：
  python src/predict_image.py --source data/demo.jpg --openvino
  python src/predict_video.py --source data/demo.mp4 --openvino
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--int8", action="store_true", help="INT8 量化（需要 openvino[nncf]）")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"找不到 {args.weights}，先把训练好的 best.pt 拷贝到 models/ 下")
    model = YOLO(args.weights)
    path = model.export(format="openvino", imgsz=args.imgsz, half=False, int8=args.int8)
    print(f"导出完成: {path}")
    print("之后用 --openvino 参数即可调用加速模型")


if __name__ == "__main__":
    main()
