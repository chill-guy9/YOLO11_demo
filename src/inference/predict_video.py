"""
视频/摄像头推理脚本（演示"能够识别视频"）
==========================================
用法：
  python src/predict_video.py --source data/demo.mp4            # 视频文件，保存结果
  python src/predict_video.py --source data/demo.mp4 --show     # 弹窗实时观看（ESC 退出）
  python src/predict_video.py --source 0                        # 摄像头
  python src/predict_video.py --source data/demo.mp4 --stride 2 # 每 2 帧推理 1 次（CPU 慢时保流畅）

说明：
  - --stride N 表示每 N 帧才做一次推理，中间帧复用上一次的检测框画上去，
    推理 FPS 不变但输出视频更流畅；FPS 计数只统计真实推理
  - 类别名建议用英文：cv2.putText 无法渲染中文（会画成 ???）
  - 结果视频固定写到 runs/video/result.mp4（纯英文路径，避免 OpenCV 编码器踩中文路径的坑）
"""
import argparse
import time
from pathlib import Path

import cv2

from predict_image import find_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="视频文件路径或摄像头序号（0/1）")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.4, help="NMS IoU 阈值，越小抑制越强")
    ap.add_argument("--max-det", type=int, default=1, help="每帧最多保留几个目标；演示默认 1 最干净")
    ap.add_argument("--stride", type=int, default=1, help="每几帧推理一次")
    ap.add_argument("--show", action="store_true", help="弹窗显示（默认只保存文件）")
    ap.add_argument("--openvino", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = args.weights or find_weights(args.openvino)
    print(f"加载模型: {weights}")
    model = YOLO(weights)

    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频源: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path("runs/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (width, height),
    )
    print(f"输入 {width}x{height} @ {fps_src:.1f}fps -> runs/video/result.mp4（Ctrl+C 或 ESC 中断）")

    last_boxes = []  # [(cls_name, conf, x1, y1, x2, y2)]
    frame_idx = 0
    n_infer = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride == 0:
                res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, max_det=args.max_det,
                                    device="cpu", verbose=False)[0]
                last_boxes = []
                if res.boxes is not None:
                    for b in res.boxes:
                        last_boxes.append(
                            (res.names[int(b.cls)], float(b.conf), *map(float, b.xyxy[0]))
                        )
                n_infer += 1
            for name, conf, x1, y1, x2, y2 in last_boxes:
                color = (0, 114, 255)  # BGR 橙
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 8),
                              (int(x1) + tw, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elapsed = time.time() - t0
            fps = n_infer / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"infer {fps:.1f} FPS | stride={args.stride}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            writer.write(frame)
            if args.show:
                cv2.imshow("YOLO11 Landmark", frame)
                if cv2.waitKey(1) == 27:  # ESC
                    break
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
        total = time.time() - t0
        print(f"处理 {frame_idx} 帧，其中推理 {n_infer} 帧，平均推理 {n_infer / max(total, 1e-6):.1f} FPS")
        print(f"结果视频: {out_dir / 'result.mp4'}")


if __name__ == "__main__":
    main()
