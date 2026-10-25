"""
X-AnyLabeling <-> YOLO 标注格式互转（解决"打开文件夹看不到框"）
================================================================
背景：X-AnyLabeling 打开目录时只自动加载它自己的 JSON（<图片名>.json，labelme 风格），
      不读 YOLO 的 .txt —— 直接打开 images/ 目录看不到任何框，标题显示 [未标注]。
      官方的"导入 YOLO"入口也能用，但标注保存后又会变回 JSON，训练前仍需转回 txt，
      所以这里直接用 JSON 做中间格式，两边都省事。

用法（在项目根目录执行）：
  python src/xlabel_bridge.py to_json   # YOLO txt -> JSON（X-AnyLabeling 重开目录即可见框）
  python src/xlabel_bridge.py to_yolo   # 人工修完 -> JSON 转回 YOLO txt（训练用，覆盖原 txt）

说明：
  - to_json 只为"有框"的图生成 JSON；没有框的图不生成，
    在 X-AnyLabeling 里会显示 [未标注]，正好就是需要你补框的那批（清单见 _needs_manual.txt）
  - to_yolo 会覆盖 data/labeled/labels/ 下的同名 .txt，并打印每类框数
"""
import argparse
import json
from pathlib import Path

IMG_DIR = Path("data/labeled/images")
LBL_DIR = Path("data/labeled/labels")
CLASSES_FILE = Path("data/classes.txt")


def load_classes():
    names = []
    for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines():
        n = line.split("#", 1)[0].strip()
        if n:
            names.append(n)
    return names


def to_json():
    from PIL import Image

    names = load_classes()
    written = skipped = 0
    for lbl in sorted(LBL_DIR.glob("*.txt")):
        lines = [l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            skipped += 1
            continue
        img_path = IMG_DIR / (lbl.stem + ".jpg")
        if not img_path.exists():
            print(f"[跳过] 找不到图片: {img_path.name}")
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        shapes = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = max(0.0, min(1.0, cx - bw / 2)) * w
            y1 = max(0.0, min(1.0, cy - bh / 2)) * h
            x2 = max(0.0, min(1.0, cx + bw / 2)) * w
            y2 = max(0.0, min(1.0, cy + bh / 2)) * h
            shapes.append(
                {
                    "label": names[cid],
                    "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": [],
                }
            )
        data = {
            "version": "2.4.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        (IMG_DIR / (lbl.stem + ".json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1
    print(f"[to_json] 生成 {written} 个 JSON（有框的图）；跳过 {skipped} 个空标注图（待你补框）")
    print("下一步：关闭 X-AnyLabeling 后重新打开目录 data/labeled/images/，即可看到红框")


def to_yolo():
    names = load_classes()
    name_to_id = {n: i for i, n in enumerate(names)}
    jsons = sorted(IMG_DIR.glob("*.json"))
    total_boxes = 0
    per_class = {n: 0 for n in names}
    empty = 0
    for jf in jsons:
        data = json.loads(jf.read_text(encoding="utf-8"))
        lines = []
        for sh in data.get("shapes", []):
            if sh.get("shape_type") not in ("rectangle", "rotation"):
                continue
            pts = sh.get("points", [])
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w_img = data.get("imageWidth") or 0
            h_img = data.get("imageHeight") or 0
            if not w_img or not h_img:
                from PIL import Image

                with Image.open(IMG_DIR / (jf.stem + ".jpg")) as im:
                    w_img, h_img = im.size
            x1, x2 = max(0, min(xs)), min(w_img, max(xs))
            y1, y2 = max(0, min(ys)), min(h_img, max(ys))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            label = sh.get("label")
            if label not in name_to_id:
                print(f"[跳过] 未知类别 '{label}' in {jf.name}")
                continue
            cid = name_to_id[label]
            lines.append(
                f"{cid} {(x1 + bw / 2) / w_img:.6f} {(y1 + bh / 2) / h_img:.6f} "
                f"{bw / w_img:.6f} {bh / h_img:.6f}"
            )
            per_class[label] += 1
            total_boxes += 1
        (LBL_DIR / (jf.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        if not lines:
            empty += 1
    print(f"[to_yolo] 处理 {len(jsons)} 个 JSON -> YOLO txt")
    print(f"  总框数 {total_boxes}；空标注 {empty} 张")
    print("  各类框数:", {k: v for k, v in per_class.items() if v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["to_json", "to_yolo"])
    args = ap.parse_args()
    to_json() if args.mode == "to_json" else to_yolo()
