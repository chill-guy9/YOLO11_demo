"""
ONNX 导出变体实测：静态基线 / 动态尺寸 / NMS 入图 / FP16
======================================================
对同一份 best.pt 分别导出，逐一测：文件体积、图输入输出、test 集 mAP、
CPU 端到端延迟、能否在 CPU 上被 ONNX Runtime 正常加载。

产物全部放在 compare/variants/ 下，**不会覆盖主产物 weights/best.onnx**。
用法： python src/train_eval/benchmark_onnx_variants.py
"""
import json
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

import onnx
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/detect/landmark_v5_1025"
PT = RUN / "weights" / "best.pt"
BASE_ONNX = RUN / "weights" / "best.onnx"
OUT = RUN / "compare" / "variants"
DATA = ROOT / "data.yaml"
TEST_IMGS = sorted((ROOT / "data/dataset/images/test").glob("*.jpg"))[:20]
IMGSZ = 320


def io_info(path: Path) -> dict:
    m = onnx.load(str(path))

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in v.type.tensor_type.shape.dim]

    dtypes = Counter(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(t.data_type)).name
                     for t in m.graph.initializer)
    return {
        "inputs": [{"name": i.name, "shape": dims(i)} for i in m.graph.input],
        "outputs": [{"name": o.name, "shape": dims(o)} for o in m.graph.output],
        "nodes": len(m.graph.node),
        "weight_dtypes": dict(dtypes),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def evaluate(path: Path, tag: str) -> dict:
    """val 测 mAP，predict 测端到端延迟；任一步失败都记录原因而不中断整批。"""
    res = {}
    try:
        model = YOLO(str(path))
    except Exception as e:
        return {"load_error": f"{type(e).__name__}: {e}"}

    try:
        r = model.val(data=str(DATA), split="test", imgsz=IMGSZ, batch=1,
                      rect=False, device="cpu", plots=False, verbose=False,
                      project=str(OUT), name=f"val_{tag}")
        res["mAP50"] = round(float(r.box.map50), 4)
        res["mAP50-95"] = round(float(r.box.map), 4)
        res["P"] = round(float(r.box.mp), 4)
        res["R"] = round(float(r.box.mr), 4)
    except Exception as e:
        res["val_error"] = f"{type(e).__name__}: {e}"

    try:
        for p in TEST_IMGS[:3]:
            model.predict(str(p), imgsz=IMGSZ, device="cpu", verbose=False)
        times = []
        for _ in range(2):
            for p in TEST_IMGS:
                t0 = time.perf_counter()
                model.predict(str(p), imgsz=IMGSZ, device="cpu", verbose=False)
                times.append((time.perf_counter() - t0) * 1000)
        res["median_ms"] = round(statistics.median(times), 2)
        res["fps"] = round(1000 / statistics.median(times), 1)
        res["n"] = len(times)
    except Exception as e:
        res["speed_error"] = f"{type(e).__name__}: {e}"
    return res


def export_variant(name: str, extra: dict) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    tmp_pt = d / "model.pt"
    shutil.copy2(PT, tmp_pt)
    t0 = time.perf_counter()
    kwargs = dict(format="onnx", imgsz=IMGSZ, opset=17, simplify=True,
                  dynamic=False, nms=False, half=False, device="cpu")
    kwargs.update(extra)
    exported = YOLO(str(tmp_pt)).export(**kwargs)
    return Path(exported), round((time.perf_counter() - t0) * 1000, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASE_ONNX.exists():
        raise SystemExit("缺少主产物 best.onnx，请先运行 src/inference/export_onnx.py")

    specs = [
        ("fp32_static", None),
        ("fp32_dynamic", {"dynamic": True}),
        ("fp32_nms", {"nms": True}),
        ("fp16_static", {"half": True}),
    ]
    rows = {}
    for name, extra in specs:
        print(f"[VARIANT] {name} 开始")
        row = {}
        try:
            if extra is None:
                onnx_path, export_ms = BASE_ONNX, 0.0
            else:
                onnx_path, export_ms = export_variant(name, extra)
            row["export_ms"] = export_ms
            row["file"] = str(onnx_path.relative_to(ROOT)).replace("\\", "/")
            row.update(io_info(onnx_path))
            row.update(evaluate(onnx_path, name))
        except Exception as e:
            row["export_error"] = f"{type(e).__name__}: {e}"
        rows[name] = row
        print(f"[VARIANT] {name}: {json.dumps(row, ensure_ascii=False)}")

    (OUT / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 导出变体对比 =====")
    print("| 变体 | 体积 MB | 输入 | 输出 | 权重精度 | mAP@50 | 中位延迟 ms | 备注 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        inp = r.get("inputs", [{}])[0].get("shape", "-")
        outp = r.get("outputs", [{}])[0].get("shape", "-")
        dt = "/".join(r.get("weight_dtypes", {}).keys()) or "-"
        note = ", ".join(k for k in r if k.endswith("_error")) or "正常"
        print(f"| {name} | {r.get('size_mb', '-')} | {inp} | {outp} | {dt} | "
              f"{r.get('mAP50', '-')} | {r.get('median_ms', '-')} | {note} |")
    print(f"\n结果: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
