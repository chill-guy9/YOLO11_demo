"""
生成推理演示用的视频素材（无需联网）
====================================
把测试集图片按类别顺序合成一段带运镜（Ken Burns）的 MP4，
让"视频演示"有连续画面、且每类建筑按顺序出现，方便录屏讲解。

用法：
  python src/make_demo_video.py                       # 默认 1280x720@25fps，每类 5 张，每张 2.5s
  python src/make_demo_video.py --per-class 6 --sec 3
  python src/make_demo_video.py --split test --out data/demo.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_classes(path="data/classes.txt"):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        n = line.split("#", 1)[0].strip()
        if n:
            names.append(n)
    return names


def cover_resize(img, w, h):
    """等比缩放到完全覆盖 (w,h)，多余部分随后裁掉"""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    return cv2.resize(img, (max(w, int(round(iw * s))), max(h, int(round(ih * s)))),
                      interpolation=cv2.INTER_AREA)


def read_image(path):
    """兼容中文路径读图"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def ken_burns_frames(base, W, H, n_frames, zoom_from=1.18, zoom_to=1.02, pan_dir=1):
    """从放大后的底图上取一个逐渐收紧的窗口，产生缓慢推近 + 平移的镜头感"""
    bh, bw = base.shape[:2]
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        s = zoom_from + (zoom_to - zoom_from) * t
        cw, ch = min(bw, int(round(W * s))), min(bh, int(round(H * s)))
        pan = (t - 0.5) * pan_dir * 0.7  # -0.35 ~ +0.35
        cx = bw / 2 + pan * (bw - cw) / 2
        cy = bh / 2 + pan * (bh - ch) / 2 * -0.5
        x1 = int(round(cx - cw / 2))
        y1 = int(round(cy - ch / 2))
        x1 = max(0, min(bw - cw, x1))
        y1 = max(0, min(bh - ch, y1))
        crop = base[y1:y1 + ch, x1:x1 + cw]
        yield cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="用哪个划分的图片（test 最贴近演示场景）")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--sec", type=float, default=2.5, help="每张图停留秒数")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--out", default="data/demo.mp4")
    args = ap.parse_args()

    W, H = map(int, args.size.lower().split("x"))
    names = load_classes()
    img_dir = Path(f"data/dataset/images/{args.split}")
    if not img_dir.exists():
        raise SystemExit(f"找不到图片目录: {img_dir}")

    picks = []
    for ci, name in enumerate(names):
        files = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTS and p.stem.startswith(name))
        if len(files) < args.per_class:
            print(f"[提示] {name} 只有 {len(files)} 张，少于 --per-class {args.per_class}")
        picks.extend((ci, p) for p in files[:args.per_class])
    if not picks:
        raise SystemExit("没有可用的图片")

    # OpenCV 的 VideoWriter 对中文路径不稳，先写英文临时路径再搬回
    tmp = Path("C:/tmp") / "demo_build.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter 打不开: {tmp}")

    n_per = int(args.sec * args.fps)
    gap = max(1, args.fps // 5)  # 段间黑场
    total = 0
    for k, (ci, path) in enumerate(picks):
        img = read_image(path)
        if img is None:
            print(f"[跳过] 读不出: {path}")
            continue
        base = cover_resize(img, int(W * 1.2), int(H * 1.2))
        for f in ken_burns_frames(base, W, H, n_per, pan_dir=1 if k % 2 == 0 else -1):
            writer.write(f)
            total += 1
        if k != len(picks) - 1:
            for _ in range(gap):
                writer.write(np.zeros((H, W, 3), dtype=np.uint8))
                total += 1
        print(f"[{k + 1}/{len(picks)}] {path.stem}")

    writer.release()

    # 搬到目标路径（用 shutil 走 Python 层，绕开 OpenCV 的中文路径限制）
    import shutil
    shutil.move(str(tmp), str(out))
    print(f"\n完成: {out.resolve()}")
    print(f"  {len(picks)} 段 × {args.sec}s + 转场 → {total} 帧 / {total / args.fps:.1f}s @ {W}x{H}")


if __name__ == "__main__":
    main()
