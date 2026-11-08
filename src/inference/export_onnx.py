"""
把 YOLO11 训练好的 .pt 权重导出为 ONNX（FP32，默认静态尺寸）
=============================================================
与训练参数保持一致：imgsz=320（见 runs/detect/landmark_v5_1025/args.yaml）。

用法：
  python src/inference/export_onnx.py
  python src/inference/export_onnx.py --imgsz 640 --dynamic
  python src/inference/export_onnx.py --weights runs/detect/xxx/weights/best.pt

产物默认落在 .pt 同目录下（best.onnx）。导出后推理：
  from ultralytics import YOLO
  YOLO("runs/detect/landmark_v5_1025/weights/best.onnx").predict("xxx.jpg")
"""
import argparse
from pathlib import Path

import onnx
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PT = ROOT / "runs" / "detect" / "landmark_v5_1025" / "weights" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11 .pt -> .onnx (FP32)")
    ap.add_argument("--weights", default=str(DEFAULT_PT), help="训练好的 .pt 权重路径")
    ap.add_argument("--imgsz", type=int, default=320, help="输入尺寸，需与训练/评估一致")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    ap.add_argument("--dynamic", action="store_true", help="动态 batch/尺寸（部署多分辨率时用）")
    ap.add_argument("--nms", action="store_true", help="把 NMS 一起写进图（默认由图外处理）")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", help="跳过 onnxslim 化简")
    ap.set_defaults(simplify=True)
    args = ap.parse_args()

    pt = Path(args.weights)
    if not pt.exists():
        raise SystemExit(f"找不到权重: {pt}")

    print(f"源权重 : {pt}  ({pt.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"导出参数: format=onnx, imgsz={args.imgsz}, opset={args.opset}, "
          f"dynamic={args.dynamic}, simplify={args.simplify}, nms={args.nms}, FP32")

    model = YOLO(str(pt))
    onnx_path = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=args.nms,
        half=False,
        int8=False,
        device="cpu",
    ))

    # 结构校验 + 图信息
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("\n[校验] onnx.checker 通过")
    print(f"输出文件: {onnx_path}  ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"IR / opset: {onnx_model.ir_version} / "
          f"{[(i.domain or 'ai.onnx', i.version) for i in onnx_model.opset_import]}")
    print(f"节点数  : {len(onnx_model.graph.node)}")
    print("输入:")
    for x in onnx_model.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in x.type.tensor_type.shape.dim]
        print(f"  {x.name}: {dims}")
    print("输出:")
    for y in onnx_model.graph.output:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in y.type.tensor_type.shape.dim]
        print(f"  {y.name}: {dims}")
    meta = {p.key: p.value for p in onnx_model.metadata_props}
    if meta:
        print("元数据:", {k: (v[:80] + "..." if len(v) > 80 else v) for k, v in meta.items()})


if __name__ == "__main__":
    main()
