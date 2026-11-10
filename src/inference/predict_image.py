"""
图像推理 —— 「把图片放进放置位置 → 自动出画框图」

默认工作流（PyCharm 直接点运行，不用加任何参数）：
  1. 把待识别图片放进  data/predict_inputs/
  2. 运行本脚本
  3. 画框结果出现在  runs/detect/predict_inputs/<微秒时间戳>/
  4. 原图不动、旧结果不覆盖、只处理本次「新增」的图片

行为约定：
  - 只「读」放置位置里的图片，永不改写 / 删除 / 重命名
  - 每次运行开一个独立时间戳目录，同批重名自动加序号
  - 已处理图片记在 .processed.json（文件名|大小|修改时间），重复运行不重复出图
  - 无新增图片时不生成任何文件；--reprocess 可强制重跑全部

老用法（--source 指定单张图片或文件夹）：
  python src/inference/predict_image.py --source data/demo_imgs/
  python src/inference/predict_image.py --source eiffel-tower-test.jpg --onnx
  python src/inference/predict_image.py --source eiffel-tower-test.jpg --openvino

常用参数：
  --input-dir / --out-root   换「图片放置位置」/「画框结果根目录」
  --reprocess                忽略已处理记录，重跑放置位置里所有图片（仍写入新目录）
  --no-save                  只打印识别文字，不写画框图
  --onnx                     把显式指定的 .pt 换成同目录 .onnx
  --openvino                 使用 models/best_openvino_model（存在时）
  --weights / --imgsz / --conf / --iou / --max-det   模型与推理参数
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================================================================== 配置
ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "predict_inputs"            # 图片放置位置
OUT_ROOT = ROOT / "runs" / "detect" / "predict_inputs"   # 画框结果根目录
DEFAULT_IMG = ROOT / "statue_of_liberty.webp"            # 放置位置为空时的演示图
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "landmark_v5_1025" / "weights" / "best.onnx"
OPENVINO_DIR = ROOT / "models" / "best_openvino_model"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SAFE_OUT_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEP = "-" * 72


class Mode:
    """本次运行的图片来源，决定是否更新 .processed.json。"""
    SOURCE = "source"   # --source 老用法
    INBOX = "inbox"     # 放置位置里的新增图片
    DEMO = "demo"       # 放置位置为空时的演示图
    IDLE = "idle"       # 没有新图，什么都不做


@dataclass
class RunPlan:
    paths: list[Path]
    mode: str
    note: str = ""      # 需要额外提示用户的一行文字


# ===================================================================== 权重与后端
def find_latest_pt() -> Path | None:
    """默认 ONNX 不在时（例如目录迁移），回退到最近一次训练的 best.pt。"""
    fixed = ROOT / "models" / "best.pt"
    if fixed.is_file():
        return fixed
    cands = sorted((ROOT / "runs").rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def backend_of(weights: Path) -> str:
    if weights.is_dir():
        return "OpenVINO"
    if weights.suffix.lower() == ".onnx":
        return "ONNX Runtime"
    if weights.suffix.lower() == ".pt":
        return "PyTorch"
    return "未知后端"


def resolve_weights(args: argparse.Namespace) -> tuple[str, str]:
    """按命令行挑权重，返回 (权重路径, 后端标签)。

    优先级：显式 --weights > --openvino > 默认 ONNX。
    默认 ONNX 缺失时回退最近一次训练的 best.pt；显式路径写错则直接报错。
    """
    if args.weights:
        weights = Path(args.weights)
    elif args.openvino and OPENVINO_DIR.is_dir():
        weights = OPENVINO_DIR
    else:
        if args.openvino:
            print(f"[警告] 未找到 {OPENVINO_DIR}，回退默认 ONNX 权重")
        weights = DEFAULT_WEIGHTS

    # --onnx：把显式指定的 .pt 换成同目录 .onnx
    if args.onnx and weights.suffix.lower() == ".pt":
        weights = weights.with_suffix(".onnx")
        if not weights.is_file():
            raise SystemExit(f"找不到 {weights}，请先运行 src/inference/export_onnx.py")

    if not weights.exists():
        if args.weights:
            raise SystemExit(f"指定的权重不存在：{weights}")
        print(f"[提示] 权重不存在：{weights}，尝试查找 best.pt")
        fallback = find_latest_pt()
        if fallback is None:
            raise SystemExit("找不到权重文件：请先训练，或把权重放到 models/ 下，或用 --weights 指定")
        print(f"[提示] 使用权重: {fallback}")
        weights = fallback

    return str(weights), backend_of(weights)


# ===================================================================== 图片读写
def imread_unicode(p: Path):
    """只读打开图片，兼容 Windows 中文路径（不用 cv2.imread）；失败返回 None。"""
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


# ===================================================================== 已处理记录
def fingerprint(p: Path) -> str:
    """内容指纹：同名文件被换成新内容，也算「新加入一张图」。"""
    st = p.stat()
    return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"


def load_processed(log: Path) -> set[str]:
    if not log.exists():
        return set()
    try:
        return set(json.loads(log.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def save_processed(log: Path, keys: set[str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(sorted(keys), ensure_ascii=False, indent=2), encoding="utf-8")


# ===================================================================== 画框图输出
def make_run_dir(out_root: Path) -> Path:
    """每次运行开一个带微秒时间戳的新目录 —— 旧结果永不被覆盖。"""
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def out_name(p: Path, used: set[str]) -> str:
    """同一次运行里若出现重名，自动加序号，绝不互相覆盖。"""
    ext = p.suffix.lower() if p.suffix.lower() in SAFE_OUT_EXTS else ".jpg"
    base, i = p.stem, 1
    name = f"{base}{ext}"
    while name.lower() in used:
        i += 1
        name = f"{base}_{i}{ext}"
    used.add(name.lower())
    return name


def save_plot(result, out_path: Path) -> bool:
    """把画框图写到指定路径；result.plot() 返回原图副本，不会改动原文件。"""
    im = result.plot()
    suffix = out_path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if suffix in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(suffix, im, params)
    if ok:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(out_path))  # tofile 规避中文路径写盘问题
    return bool(ok)


# ===================================================================== 运行计划与推理
def plan_run(args: argparse.Namespace, input_dir: Path, done: set[str]) -> RunPlan:
    """决定本次处理哪些图、属于哪种模式；不打印、不写文件。"""
    if args.source:
        src = Path(args.source)
        paths = list_images(src) if src.is_dir() else [src]
        return RunPlan(paths, Mode.SOURCE)

    all_imgs = list_images(input_dir)
    fresh = [p for p in all_imgs if fingerprint(p) not in done]
    if fresh:
        return RunPlan(fresh, Mode.INBOX)
    if not all_imgs and DEFAULT_IMG.exists():
        return RunPlan(
            [DEFAULT_IMG], Mode.DEMO,
            note=f"[提示] 放置位置还是空的，本次先跑项目自带演示图 {DEFAULT_IMG.name}",
        )
    return RunPlan([], Mode.IDLE)


def predict_one(model, im0, args):
    """只读 ndarray 推理；输出路径由我们自己决定，故 save=False。"""
    return model.predict(
        source=im0,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det if args.max_det > 0 else 300,
        device="cpu",
        save=False,
        verbose=False,
    )[0]


def report(result, name: str) -> None:
    """把一张图的检测框打印成可读文字。"""
    if result.boxes is None or len(result.boxes) == 0:
        print(f"{name}: 未检测到目标")
        return
    for b in result.boxes:
        cls = result.names[int(b.cls)]
        conf = float(b.conf)
        xyxy = [round(v, 1) for v in b.xyxy[0].tolist()]
        print(f"{name}: {cls} (conf={conf:.2f}) box={xyxy}")


# ===================================================================== 命令行
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="把图片放进放置位置 → 自动出画框图（默认 ONNX Runtime，直接点运行即可）")
    ap.add_argument("--source", default=None,
                    help="老用法：单张图片或图片文件夹；不填则走「图片放置位置」模式")
    ap.add_argument("--weights", default=None,
                    help="权重路径；留空按后端自动选（默认 v5 的 best.onnx，--openvino 时找 OpenVINO 目录）")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.4, help="NMS IoU 阈值，越小抑制越强")
    ap.add_argument("--max-det", type=int, default=0, help="每图最多保留几个目标；0 = 使用底层默认 300")
    ap.add_argument("--input-dir", default=str(INPUT_DIR), help="★ 图片放置位置")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="★ 画框结果根目录")
    ap.add_argument("--reprocess", action="store_true",
                    help="忽略已处理记录，重跑放置位置里所有图片（仍写入新目录）")
    ap.add_argument("--no-save", action="store_true", help="只打印识别结果，不写画框图")
    ap.add_argument("--save", action="store_true",
                    help="兼容旧命令；放置位置模式默认就保存，无需再加此参数")
    ap.add_argument("--onnx", action="store_true",
                    help="用 ONNX Runtime 推理（把显式指定的 .pt 换成同目录 .onnx）")
    ap.add_argument("--openvino", action="store_true", help="使用 models/best_openvino_model（存在时）")
    return ap.parse_args()


# ===================================================================== 主流程
def run(args: argparse.Namespace) -> None:
    weights, backend = resolve_weights(args)
    print(f"加载模型: {weights} [{backend}]")
    model = YOLO(weights)

    input_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    log = out_root / ".processed.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    print(f"[图片放置位置] {input_dir}")
    print("  把要识别的图片放进这个文件夹，重新运行即可；原图和旧结果都不会被覆盖。")

    done = set() if args.reprocess else load_processed(log)
    plan = plan_run(args, input_dir, done)
    if plan.note:
        print(plan.note)

    if plan.mode == Mode.IDLE:
        print("  当前没有「新增」图片（放过的图都已处理过），本次不生成任何文件，旧结果保持原样。")
        print("  → 再往放置位置放一张新图片，重新运行即可生成新的画框图。")
        print("  → 想重跑旧图：加 --reprocess。")
        return
    if not plan.paths:
        print("放置位置里没有图片，也没有可用的演示图，本次没有可处理的内容。")
        return

    run_dir = make_run_dir(out_root)
    print(f"[画框结果位置] {run_dir}")
    print(f"本次待处理 {len(plan.paths)} 张 | 模式={plan.mode} | 保存={'关' if args.no_save else '开'}")
    print(SEP)

    used_names: set[str] = set()
    newly_done: set[str] = set()
    for p in plan.paths:
        im0 = imread_unicode(p)
        if im0 is None:
            print(f"{p.name}: 读取失败，跳过")
            continue
        result = predict_one(model, im0, args)
        report(result, p.name)
        if not args.no_save:
            out_path = run_dir / out_name(p, used_names)
            if save_plot(result, out_path):
                print(f"  └─ 已保存画框图: {out_path}")
            else:
                print(f"  └─ 画框图编码失败: {out_path.name}")
        if plan.mode == Mode.INBOX:
            newly_done.add(fingerprint(p))

    if plan.mode == Mode.INBOX and newly_done:
        save_processed(log, done | newly_done)
    print(SEP)
    print(f"完成。原图保持原样：{input_dir}")
    print(f"本次结果独立存放，之前的结果未被覆盖：{run_dir}")
    if args.no_save:
        print("  （--no-save：本次未写画框图）")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
