"""
纯 ONNX Runtime 推理 —— NMS 入图版（`--nms` 导出的 ONNX）
========================================================
图里已经包含 NMS，输出直接是 (x1,y1,x2,y2,conf,cls)，
所以只需要 letterbox + BGR→RGB + conf 过滤 + 坐标还原，**不用自己写 NMS**。

不 import torch / ultralytics，只依赖 numpy + cv2 + onnxruntime。
预处理函数复用 predict_onnx_native.py（同为纯 numpy 实现）。

用法：
  python src/inference/predict_onnx_nms_native.py
  python src/inference/predict_onnx_nms_native.py --source data/dataset/images/test --save
"""
import argparse
import time
from pathlib import Path

import numpy as np

from predict_onnx_native import (  # 同为纯 numpy/cv2 实现，不含 torch/ultralytics
    IMG_EXTS,
    ROOT,
    imread_unicode,
    letterbox,
    load_session,
    save_image,
)

DEFAULT_ONNX = ROOT / "runs/detect/landmark_v5_1025/compare/variants/fp32_nms/model.onnx"
DEFAULT_IMG = ROOT / "eiffel-tower-test.jpg"
OUT_DIR = ROOT / "runs/detect/onnx_nms_native"


def detect(sess, names: dict, imgsz: tuple, im0: np.ndarray,
           conf_thres: float = 0.25, max_det: int = 300):
    """返回 [{'name','conf','box'}]，坐标已还原到原图。"""
    size = int(imgsz[0])
    im, r, pad = letterbox(im0, size)
    x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    t0 = time.perf_counter()
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    d = pred[0]  # (300, 6) = xyxy(输入尺度), conf, cls；无效行全 0
    d = d[d[:, 4] > conf_thres][:max_det]
    if d.shape[0] == 0:
        return [], infer_ms

    boxes = d[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im0.shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im0.shape[0])
    dets = [{"name": names.get(int(c), str(c)),
             "conf": float(s),
             "box": [round(float(v), 1) for v in b]}
            for b, s, c in zip(boxes, d[:, 4], d[:, 5])]
    return dets, infer_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="纯 ONNX Runtime 推理（NMS 已在图内）")
    ap.add_argument("--source", default=str(DEFAULT_IMG))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--save", action="store_true", help="保存画框图到 runs/detect/onnx_nms_native/")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"找不到 {onnx_path}，请先运行 src/train_eval/benchmark_onnx_variants.py")
    sess, names, imgsz = load_session(onnx_path)
    print(f"模型: {onnx_path} | imgsz={imgsz} | 类别 {len(names)} | "
          f"输出 shape={sess.get_outputs()[0].shape}")

    src = Path(args.source)
    paths = ([p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXTS]
             if src.is_dir() else [src])
    total_ms = 0.0
    for p in paths:
        im0 = imread_unicode(p)
        if im0 is None:
            print(f"{p.name}: 读取失败")
            continue
        dets, infer_ms = detect(sess, names, imgsz, im0, args.conf, args.max_det)
        total_ms += infer_ms
        if not dets:
            print(f"{p.name}: 未检测到目标 ({infer_ms:.1f} ms)")
        for d in dets:
            print(f"{p.name}: {d['name']} conf={d['conf']:.3f} box={d['box']} "
                  f"({infer_ms:.1f} ms)")
        if args.save:
            save_image(im0.copy(), dets, OUT_DIR / p.name)
    if paths:
        print(f"共 {len(paths)} 张，纯推理平均 {total_ms / len(paths):.1f} ms/张")


if __name__ == "__main__":
    main()
