"""
CPU 训练入口（YOLO11n @ 320px，COCO 预训练权重迁移学习）
=========================================================
针对本机（AMD Ryzen 7 7840H, 8核16线程, 无 GPU）的默认配置：
  - yolo11n：最小模型（260 万参数），CPU 上训练/推理最快
  - imgsz=320：比 640 少 4 倍像素量，训练时间约缩到 1/4；建筑是大目标，320 足够
  - freeze=10：冻结 backbone，训练提速约 1/3，小数据集上精度损失很小
  - batch=16：29GB 内存完全撑得住；显存(内存)紧张可降到 8

用法（在项目根目录 D:\桌面\YOLO训练识别 下运行）：
  python src/train_eval/train.py                                  # 默认配置直接开跑
  python src/train_eval/train.py --epochs 100 --imgsz 320 --batch 16 --freeze 10
  python src/train_eval/train.py --model yolo11s --imgsz 416 --freeze 0   # 时间充裕时试更大模型
  先跑 1 个 epoch 测速：python src/train_eval/train.py --epochs 1 --name test_speed
  ⚠️ --name 每次必须用新名字（旧结果：landmark5_v3_wallfix / landmark_v4_480 不可复用）
"""
import argparse
import os
import time
from pathlib import Path

# Windows 上 PyTorch 与 conda 的 MKL 各带一份 OpenMP 运行时（libiomp5md.dll），
# 同时初始化会触发 "OMP Error #15" 并中止程序。放行重复加载是 Windows 下的通行做法，
# 必须在 import torch 之前设置。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolo11n", help="yolo11n / yolo11s / ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10, help="冻结前 n 层；0 表示不冻结")
    ap.add_argument(
        "--patience",
        type=int,
        default=30,
        help="早停耐心值；用 cos_lr 时建议设 >= epochs（让余弦退火跑完），如 --patience 100",
    )
    ap.add_argument("--workers", type=int, default=0, help="Windows 先用 0；加载慢再试 2/4")
    ap.add_argument(
        "--cache",
        default="ram",
        choices=["ram", "disk", "none"],
        help="图片缓存：ram 最快（本数据集 300 余张，解码后约 1.5GB 内存）；none 关闭",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="landmark_yolo11n")
    args = ap.parse_args()

    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    print(f"PyTorch {torch.__version__} | 设备 {args.device} | 计算线程 {n_threads}")

    from ultralytics import YOLO

    weights = args.model if args.model.endswith(".pt") else args.model + ".pt"
    model = YOLO(weights)  # 首次运行自动下载 COCO 预训练权重（约 5.6MB）

    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        freeze=args.freeze,
        patience=args.patience,  # 无提升早停（cos_lr 时建议 >= epochs）
        cos_lr=True,       # 余弦退火，小数据集上更稳
        mosaic=1.0,        # 小数据集开满 Mosaic 增强
        plots=True,        # 输出 PR 曲线、混淆矩阵等
        # 必须传绝对路径：ultralytics 会把相对 project 再拼一层 settings.runs_dir，
        # 传 "runs/detect" 会变成 runs/runs/detect/...（实测踩过）
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
    )
    print(f"训练完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")

    # 评估：有独立 test 集就用 test，否则退回 val
    split = "test" if (Path("data/dataset/images/test")).exists() else "val"
    metrics = model.val(split=split)
    print(f"[{split}] mAP@50={metrics.box.map50:.3f}  mAP@50-95={metrics.box.map:.3f}")
    print(f"权重位置: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
