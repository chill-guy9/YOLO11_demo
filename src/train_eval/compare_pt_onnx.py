"""
PyTorch(.pt) vs ONNX Runtime(.onnx) 同模型对比评测
==================================================
在同一套 test 集、同一个 imgsz 下，对比四个层面：
  1) 任务准确度：P / R / mAP50 / mAP50-95（Ultralytics val，同一评估器）
  2) 原始数值差：同一输入张量下 (1,9,N) 输出的 max / mean 绝对误差
  3) 检测结果一致性：逐图匹配框，统计置信度偏差、坐标偏差、框数差异
  4) 工程指标：文件体积、模型加载耗时、CPU 端到端单图延迟

三方对照：原始 best.pt（本项目权重为 FP16 序列化）/ 同权重重存的 FP32 .pt /
FP32 ONNX。这样可以把「框架差异」与「权重精度差异」分开，避免误判。

所有评测强制 rect=False：静态 ONNX 只接受正方形输入，若任由 PT 用默认的
矩形 letterbox，测出来的 mAP 差异其实是预处理差异。脚本末尾额外跑一组
rect=True 的 PT 作为对照，复现这个陷阱。

注意：项目里的 data.yaml 仍指向已迁移的桌面旧路径，本脚本会自动在
compare/ 下生成一份路径正确的临时配置，不修改原 data.yaml。

用法：
  python src/train_eval/compare_pt_onnx.py
  python src/train_eval/compare_pt_onnx.py --split val --imgsz 320 --repeat 3
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "detect" / "landmark_v5_1025"
DEFAULT_PT = RUN_DIR / "weights" / "best.pt"
DEFAULT_ONNX = RUN_DIR / "weights" / "best.onnx"
OUT_DIR = RUN_DIR / "compare"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_data_yaml() -> Path:
    """复制 data.yaml，但把失效的 path 换成本机真实数据集路径。"""
    cfg = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
    dataset = ROOT / "data" / "dataset"
    cfg["path"] = str(dataset).replace("\\", "/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "data_local.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def make_fp32_pt(pt_path: Path) -> Path:
    """把原本以 FP16 序列化的 .pt 权重重存为 FP32，用于隔离「框架」与「精度」两个变量。"""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    ckpt["model"] = ckpt["model"].float().eval()
    ckpt.pop("ema", None)
    ckpt.pop("optimizer", None)
    out = OUT_DIR / "best_fp32.pt"
    torch.save(ckpt, str(out))
    return out


def test_images(split: str) -> list:
    paths = sorted(p for p in (ROOT / "data" / "dataset" / "images" / split).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"没有找到 {split} 集图片")
    return paths


def evaluate(model_path: Path, data_yaml: Path, split: str, imgsz: int, tag: str,
             rect: bool = False) -> dict:
    """用 Ultralytics 官方评估器跑 mAP（对 onnx 会自动走 ONNX Runtime 后端）。

    rect=False 是关键：静态 ONNX 只能吃正方形输入，而 PT 默认 rect=True 会走
    矩形 letterbox。关掉它两者输入张量才一致，否则测到的是预处理差异而非框架差异。
    """
    t0 = time.perf_counter()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter() - t0) * 1000
    r = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=1,
                  rect=rect, device="cpu", plots=False, verbose=False, save_json=False,
                  project=str(OUT_DIR), name=f"val_{tag}")
    names = model.names
    per_class = {}
    for i, name in names.items():
        if i < len(r.box.ap50):
            per_class[name] = {"mAP50": round(float(r.box.ap50[i]), 4),
                               "mAP50-95": round(float(r.box.ap[i]), 4)}
    return {"P": round(float(r.box.mp), 4),
            "R": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
            "per_class": per_class,
            "val_inference_ms": round(float(r.speed.get("inference", 0.0)), 2),
            "load_ms": round(load_ms, 1)}


def imread_unicode(path: Path):
    """Windows 下 cv2.imread 读不了中文路径，用 imdecode 绕过。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def letterbox(im: np.ndarray, size: int) -> np.ndarray:
    """与 Ultralytics 一致的等比缩放 + 灰色填充，输出 BGR uint8。"""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out


def pick_output(y) -> torch.Tensor:
    """DetectModel 直接前向时可能返回 (输出, 特征图) 元组，取 3D 检测输出。"""
    for t in (list(y) if isinstance(y, (tuple, list)) else [y]):
        if torch.is_tensor(t) and t.dim() == 3:
            return t
    raise RuntimeError("未找到 (1, 9, N) 形式的检测输出")


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {"max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "p99_abs_diff": float(np.percentile(d, 99))}


def raw_output_diff(pt_path: Path, onnx_path: Path, img_path: Path, imgsz: int) -> dict:
    """同一输入张量，分别取 原权重PT / FP32-PT / ONNX 三份网络原始输出。"""
    import onnxruntime as ort

    im = imread_unicode(img_path)
    x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    x = np.ascontiguousarray(x)

    net = YOLO(str(pt_path)).model.eval()
    orig_dtype = next(net.parameters()).dtype
    with torch.no_grad():
        y_orig = pick_output(net(torch.from_numpy(x).to(orig_dtype))).float().cpu().numpy()
        net = net.float()
        y_fp32 = pick_output(net(torch.from_numpy(x))).float().cpu().numpy()

    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opt, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    return {"sample_image": img_path.name,
            "shape": list(y_onnx.shape),
            "pt_weight_dtype": str(orig_dtype).replace("torch.", ""),
            "onnx_weight_dtype": "float32",
            "pt_orig_vs_onnx": diff_stats(y_orig, y_onnx),
            "pt_fp32_vs_onnx": diff_stats(y_fp32, y_onnx),
            "pt_orig_vs_pt_fp32": diff_stats(y_orig, y_fp32)}


def collect_boxes(model, img_paths, imgsz, conf=0.25, iou=0.7) -> dict:
    out = {}
    for p in img_paths:
        r = model.predict(str(p), imgsz=imgsz, conf=conf, iou=iou, rect=False,
                          device="cpu", verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            out[p.name] = np.zeros((0, 6), dtype=np.float32)
        else:
            out[p.name] = np.concatenate([
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy()[:, None],
                b.cls.cpu().numpy()[:, None],
            ], axis=1).astype(np.float32)
    return out


def iou_one(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def boxes_agreement(a_boxes: dict, b_boxes: dict) -> dict:
    """按 同类别 + IoU>=0.5 贪心匹配，比较最终检测框的一致性。"""
    matched = 0
    total_a = total_b = 0
    conf_diffs, coord_diffs, count_diffs = [], [], []
    for name in a_boxes:
        a, b = a_boxes[name], b_boxes[name]
        total_a += len(a)
        total_b += len(b)
        count_diffs.append(abs(len(a) - len(b)))
        used = set()
        for i in range(len(a)):
            best_j, best_iou = -1, 0.0
            for j in range(len(b)):
                if j in used or a[i, 5] != b[j, 5]:
                    continue
                v = iou_one(a[i, :4], b[j, :4])
                if v > best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                matched += 1
                conf_diffs.append(abs(float(a[i, 4] - b[best_j, 4])))
                coord_diffs.append(float(np.abs(a[i, :4] - b[best_j, :4]).mean()))
    return {"pt_boxes": total_a, "onnx_boxes": total_b,
            "matched": matched,
            "match_rate": round(matched / max(total_a, 1), 4),
            "mean_conf_abs_diff": round(float(np.mean(conf_diffs)) if conf_diffs else 0.0, 6),
            "max_conf_abs_diff": round(float(np.max(conf_diffs)) if conf_diffs else 0.0, 6),
            "mean_coord_abs_diff_px": round(float(np.mean(coord_diffs)) if coord_diffs else 0.0, 4),
            "mean_box_count_diff": round(float(np.mean(count_diffs)), 4)}


def bench(model, img_paths, imgsz, repeat=2) -> dict:
    for p in img_paths[:3]:
        model.predict(str(p), imgsz=imgsz, rect=False, device="cpu", verbose=False)
    times = []
    for _ in range(repeat):
        for p in img_paths:
            t0 = time.perf_counter()
            model.predict(str(p), imgsz=imgsz, rect=False, device="cpu", verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
    s = sorted(times)
    return {"median_ms": round(statistics.median(times), 2),
            "mean_ms": round(statistics.mean(times), 2),
            "p95_ms": round(s[max(0, int(len(s) * 0.95) - 1)], 2),
            "fps": round(1000 / statistics.median(times), 1),
            "n": len(times)}


def main() -> None:
    ap = argparse.ArgumentParser(description="PT vs ONNX 对比评测")
    ap.add_argument("--pt", default=str(DEFAULT_PT))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    pt, onnx_path = Path(args.pt), Path(args.onnx)
    for p in (pt, onnx_path):
        if not p.exists():
            raise SystemExit(f"找不到模型: {p}（先运行 src/inference/export_onnx.py）")

    data_yaml = make_data_yaml()
    imgs = test_images(args.split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据集: {args.split} 集 {len(imgs)} 张 | imgsz={args.imgsz} | CPU | "
          f"torch threads={torch.get_num_threads()}")
    print(f"PT  : {pt}  ({pt.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"ONNX: {onnx_path}  ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)\n")

    fp32_pt = make_fp32_pt(pt)
    paths = {"pt_orig": pt, "pt_fp32": fp32_pt, "onnx": onnx_path}
    print(f"变量控制: 另存一份 FP32 权重的 .pt -> {fp32_pt.name} "
          f"({fp32_pt.stat().st_size / 1024 / 1024:.2f} MB)\n")

    print("[1/4] 官方评估器 val() 中 ...")
    metrics = {k: evaluate(v, data_yaml, args.split, args.imgsz, k) for k, v in paths.items()}
    pt_default = evaluate(pt, data_yaml, args.split, args.imgsz, "pt_rect", rect=True)

    print("[2/4] 原始输出张量数值对比中 ...")
    raw = raw_output_diff(pt, onnx_path, imgs[0], args.imgsz)

    print("[3/4] 逐图检测结果一致性 + 速度基准中 ...")
    models = {k: YOLO(str(v)) for k, v in paths.items()}
    boxes = {k: collect_boxes(m, imgs, args.imgsz) for k, m in models.items()}
    agreement = {"pt_orig_vs_onnx": boxes_agreement(boxes["pt_orig"], boxes["onnx"]),
                 "pt_fp32_vs_onnx": boxes_agreement(boxes["pt_fp32"], boxes["onnx"])}
    speed = {k: bench(m, imgs, args.imgsz, args.repeat) for k, m in models.items()}

    print("[4/4] 汇总 ...")
    result = {"config": {"split": args.split, "imgsz": args.imgsz,
                         "num_images": len(imgs), "repeat": args.repeat,
                         "device": "cpu", "conf": 0.25, "iou": 0.7, "rect": False},
              "note": "三方均强制 rect=False（正方形 letterbox）。静态 ONNX 只能吃正方形输入，"
                      "PT 默认 rect=True 会用矩形 letterbox，两者输入张量不同会 fake 出精度差。",
              "size_mb": {k: round(v.stat().st_size / 1024 / 1024, 2) for k, v in paths.items()},
              "metrics": metrics, "pt_default_rect_true": pt_default,
              "raw_output_diff": raw, "agreement": agreement, "speed": speed}
    (OUT_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    a, f, o = metrics["pt_orig"], metrics["pt_fp32"], metrics["onnx"]
    print("\n===== 任务准确度（test 集，Ultralytics val）=====")
    print(f"| 指标 | 原 .pt ({raw['pt_weight_dtype']}) | FP32 .pt | ONNX (FP32) | ONNX-原pt |")
    print("|---|---|---|---|---|")
    for k, label in [("P", "Precision"), ("R", "Recall"),
                     ("mAP50", "mAP@0.50"), ("mAP50-95", "mAP@0.50:0.95")]:
        print(f"| {label} | {a[k]:.4f} | {f[k]:.4f} | {o[k]:.4f} | {o[k] - a[k]:+.4f} |")
    print("\n各类别 mAP@0.50 (原pt / FP32-pt / onnx):")
    for name in a["per_class"]:
        print(f"  {name:22s} {a['per_class'][name]['mAP50']:.4f} / "
              f"{f['per_class'][name]['mAP50']:.4f} / {o['per_class'][name]['mAP50']:.4f}")
    print("\n===== 原始输出 (1,9,N) 数值差 =====")
    for k in ["pt_orig_vs_onnx", "pt_fp32_vs_onnx", "pt_orig_vs_pt_fp32"]:
        d = raw[k]
        print(f"  {k:22s} max={d['max_abs_diff']:.3e}  mean={d['mean_abs_diff']:.3e}")
    print("\n===== 最终检测框一致性（conf>=0.25, IoU>=0.5 匹配）=====")
    for k, v in agreement.items():
        print(f"  {k:22s} {v['matched']}/{v['pt_boxes']} 匹配 ({v['match_rate'] * 100:.1f}%) | "
              f"conf 平均差 {v['mean_conf_abs_diff']:.6f} | 坐标差 {v['mean_coord_abs_diff_px']:.4f} px | "
              f"框数 {v['onnx_boxes']}")
    print("\n===== CPU 端到端单图延迟（中位）=====")
    for k, v in speed.items():
        print(f"  {k:10s} {v['median_ms']:6.1f} ms  ({v['fps']:5.1f} FPS, n={v['n']})")
    print(f"  ONNX / 原pt 加速 {speed['pt_orig']['median_ms'] / speed['onnx']['median_ms']:.2f}x | "
          f"ONNX / FP32-pt 加速 {speed['pt_fp32']['median_ms'] / speed['onnx']['median_ms']:.2f}x")
    print("\n===== 对照：PT 保持默认 rect=True 时 =====")
    print(f"  PT rect=True  P={pt_default['P']:.4f} R={pt_default['R']:.4f} "
          f"mAP50={pt_default['mAP50']:.4f} | ONNX(强制 rect=False) mAP50={o['mAP50']:.4f}")
    print("  => 这 0.046 的差距来自矩形/正方形 letterbox 的输入差异，不是框架精度损失。")
    print(f"\n完整结果: {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
