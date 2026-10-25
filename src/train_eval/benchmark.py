"""
推理速度基准：PyTorch vs OpenVINO（CPU）
========================================
测的是端到端单图延迟（含预处理/后处理），不是纯前向。
先 warmup 若干张，再多轮取中位数，避免首帧编译/线程池抖动。

用法：
  python src/benchmark.py                              # 默认测 models/best.pt 与 OpenVINO
  python src/benchmark.py --imgs data/demo_imgs --imgsz 320 --repeat 2
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folder, limit):
    imgs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"{folder} 下没有图片")
    return imgs[:limit] if limit else imgs


def bench_one(model, imgs, imgsz, warmup=3, repeat=1):
    for p in imgs[:warmup]:
        model.predict(str(p), imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for _ in range(repeat):
        for p in imgs:
            t0 = time.perf_counter()
            model.predict(str(p), imgsz=imgsz, device="cpu", verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def stat_line(times):
    med = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    return (f"中位 {med:6.1f} ms ({1000 / med:5.1f} FPS) | "
            f"均值 {statistics.mean(times):6.1f} ms | p95 {p95:6.1f} ms | n={len(times)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--ov", default="models/best_openvino_model")
    ap.add_argument("--imgs", default="data/demo_imgs")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    imgs = collect_images(args.imgs, args.limit)
    print(f"测试图片 {len(imgs)} 张 @ imgsz={args.imgsz}，CPU 线程 {torch.get_num_threads()}")

    results = {}
    if Path(args.pt).exists():
        m = YOLO(args.pt)
        t = bench_one(m, imgs, args.imgsz, repeat=args.repeat)
        results["PyTorch"] = t
        print(f"PyTorch   : {stat_line(t)}")
    if Path(args.ov).exists():
        m = YOLO(args.ov)
        t = bench_one(m, imgs, args.imgsz, repeat=args.repeat)
        results["OpenVINO"] = t
        print(f"OpenVINO  : {stat_line(t)}")
    else:
        print(f"[跳过] 未找到 {args.ov}（先运行 src/export_openvino.py）")

    if len(results) == 2:
        speedup = statistics.median(results["PyTorch"]) / statistics.median(results["OpenVINO"])
        print(f"\nOpenVINO 相对 PyTorch 加速: {speedup:.2f}x")


if __name__ == "__main__":
    main()
