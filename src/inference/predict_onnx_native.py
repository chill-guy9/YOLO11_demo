"""
纯 ONNX Runtime 端到端推理（不 import torch / ultralytics）
==========================================================
自己实现 letterbox 预处理、xywh->xyxy 解码、按类别 NMS、坐标还原，
只依赖 numpy + opencv-python + onnxruntime，模拟真实的轻量部署环境。

用法：
  python src/inference/predict_onnx_native.py
  python src/inference/predict_onnx_native.py --source data/dataset/images/test --save
  python src/inference/predict_onnx_native.py --conf 0.25 --iou 0.7

说明：
  - 类别名、imgsz 直接读 ONNX 的 metadata_props，不另带配置文件
  - 静态导出图只接受正方形输入，因此 letterbox 用 auto=False
  - NMS 参数与 Ultralytics 对齐（conf 严格大于阈值、class offset=7680、max_det）
    默认 iou=0.4 与项目图像演示脚本一致；跑 val 复现时用 --iou 0.7
  - Windows 中文路径：读用 imdecode，写用 imencode + tofile
"""
import argparse
import ast
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "runs/detect/landmark_v5_1025/weights/best.onnx"
DEFAULT_IMG = ROOT / "eiffel-tower-test.jpg"
OUT_DIR = ROOT / "runs/detect/onnx_native"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_WH = 7680


def load_session(onnx_path: Path):
    """加载图并从元数据里取类别名与输入尺寸。"""
    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opt, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(meta.get("names", "{}"))
    imgsz = ast.literal_eval(meta.get("imgsz", "[320, 320]"))
    return sess, {int(k): v for k, v in names.items()}, tuple(int(v) for v in imgsz)


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def letterbox(im: np.ndarray, size: int):
    """与 Ultralytics LetterBox(auto=False) 一致的等比缩放 + 灰色填充。"""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if out.shape[:2] != (size, size):
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LINEAR)
    return out, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    y[:, 0], y[:, 1] = cx - w / 2, cy - h / 2
    y[:, 2], y[:, 3] = cx + w / 2, cy + h / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
        iou_thres: float, max_det: int) -> list:
    """按类别平移坐标后做标准 NMS（Ultralytics 的 class offset 做法）。"""
    shifted = boxes + classes[:, None] * MAX_WH
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(shifted[i, 0], shifted[rest, 0])
        y1 = np.maximum(shifted[i, 1], shifted[rest, 1])
        x2 = np.minimum(shifted[i, 2], shifted[rest, 2])
        y2 = np.minimum(shifted[i, 3], shifted[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        ai = (shifted[i, 2] - shifted[i, 0]) * (shifted[i, 3] - shifted[i, 1])
        ar = (shifted[rest, 2] - shifted[rest, 0]) * (shifted[rest, 3] - shifted[rest, 1])
        iou = inter / np.maximum(ai + ar - inter, 1e-12)
        order = rest[iou <= iou_thres]
    return keep


def detect(sess, names: dict, imgsz: tuple, im0: np.ndarray,
           conf_thres: float = 0.25, iou_thres: float = 0.4,
           max_det: int = 300):
    """返回 [{'name','conf','box':[x1,y1,x2,y2]}], 框坐标已还原到原图。"""
    size = int(imgsz[0])
    im, r, pad = letterbox(im0, size)
    # cv2 读入是 BGR，Ultralytics 的 preprocess 会翻转成 RGB 再归一化
    im = im[..., ::-1]
    x = np.ascontiguousarray(im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    t0 = time.perf_counter()
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    p = pred[0].T  # (N, 4+nc)
    cls_scores = p[:, 4:]
    conf = cls_scores.max(1)
    cls_id = cls_scores.argmax(1)
    filt = conf > conf_thres
    boxes = xywh2xyxy(p[filt, :4])
    scores = conf[filt]
    classes = cls_id[filt].astype(np.float32)
    if boxes.shape[0] == 0:
        return [], infer_ms

    idx = nms(boxes, scores, classes, iou_thres, max_det)
    boxes, scores, classes = boxes[idx], scores[idx], classes[idx].astype(int)

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im0.shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im0.shape[0])
    dets = [{"name": names.get(int(c), str(c)),
             "conf": float(s),
             "box": [round(float(v), 1) for v in b]}
            for b, s, c in zip(boxes, scores, classes)]
    return dets, infer_ms


def save_image(im0: np.ndarray, dets: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        color = (0, 114, 255)
        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(im0, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(im0, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
    ok, buf = cv2.imencode(out_path.suffix, im0)
    if ok:
        buf.tofile(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="纯 ONNX Runtime 推理（无 PyTorch / Ultralytics）")
    ap.add_argument("--source", default=str(DEFAULT_IMG), help="图片或图片文件夹")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--save", action="store_true", help="保存画框图到 runs/detect/onnx_native/")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"找不到 ONNX 模型: {onnx_path}")
    sess, names, imgsz = load_session(onnx_path)
    print(f"模型: {onnx_path} | imgsz={imgsz} | 类别 {len(names)} | "
          f"provider={sess.get_providers()[0]}")

    src = Path(args.source)
    paths = ([p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXTS]
             if src.is_dir() else [src])
    total_ms = 0.0
    for p in paths:
        im0 = imread_unicode(p)
        if im0 is None:
            print(f"{p.name}: 读取失败")
            continue
        dets, infer_ms = detect(sess, names, imgsz, im0,
                                args.conf, args.iou, args.max_det)
        total_ms += infer_ms
        if not dets:
            print(f"{p.name}: 未检测到目标 ({infer_ms:.1f} ms)")
        for d in dets:
            print(f"{p.name}: {d['name']} conf={d['conf']:.3f} box={d['box']} "
                  f"({infer_ms:.1f} ms)")
        if args.save:
            out = OUT_DIR / p.name
            save_image(im0.copy(), dets, out)
    if paths:
        print(f"共 {len(paths)} 张，纯推理平均 {total_ms / len(paths):.1f} ms/张")


if __name__ == "__main__":
    main()
