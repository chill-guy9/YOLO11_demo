"""
方案A 数据获取：Google Landmarks Dataset v2 (GLDv2) 定向下载
=============================================================
GLDv2 是 Google 发布的全球地标数据集：约 500 万张图、8 万+ 个地标实例，
landmark 级标注（每个 landmark_id 对应一个具体地标），
正是"识别不同标志性建筑"这个任务最匹配的公开数据集。

官方仓库: https://github.com/cvdfoundation/google-landmark

元数据文件（本机已验证可达，均无需登录）：
  train.csv                        id,url,landmark_id         约 501 MB
  train_label_to_hierarchical.csv  landmark_id,category,...    约 13 MB
  category 字段是 Wikimedia 分类页 URL，最后一段就是地标名，
  例如 .../Category:Eiffel_Tower -> "Eiffel Tower"

完整图片包是 500 个 1GB 的 tar（约 500GB），本任务不需要全下：
train.csv 里带每张图的 url，按类别挑几十张直接下载即可。

三步用法：
  1) 下载元数据（一次性，约 515MB）：
       python src/download_gldv2.py --step metadata
  2) 按名称查 landmark_id（例如先看看有哪些含 "Eiffel" 的地标）：
       python src/download_gldv2.py --step find --name Eiffel
  3) 按 landmark_id 下载图片到 data/data_raw/<类别名>/ ：
       python src/download_gldv2.py --step fetch --landmark-ids 12345,67890 --names eiffel_tower,taj_mahal --per-class 60
     （--landmark-ids 与 --names 一一对应，个数必须相同）
"""
import argparse
import csv
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

META_DIR = Path("data/gldv2_meta")
TRAIN_CSV = META_DIR / "train.csv"
HIER_CSV = META_DIR / "train_label_to_hierarchical.csv"

META_URLS = {
    TRAIN_CSV.name: "https://s3.amazonaws.com/google-landmark/metadata/train.csv",
    HIER_CSV.name: "https://s3.amazonaws.com/google-landmark/metadata/train_label_to_hierarchical.csv",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def step_metadata() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in META_URLS.items():
        dst = META_DIR / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {dst} 已存在（{dst.stat().st_size / 1e6:.1f} MB）")
            continue
        print(f"[download] {url}")
        print(f"        -> {dst}（数百 MB，请耐心等待）")
        urllib.request.urlretrieve(url, dst)
        print(f"[done] {dst}（{dst.stat().st_size / 1e6:.1f} MB）")


def step_find(keyword: str) -> None:
    """在层级元数据里按名称关键词找 landmark_id 及其图片数参考。"""
    if not HIER_CSV.exists():
        raise SystemExit(f"缺少 {HIER_CSV}，请先执行 --step metadata")
    hits = []
    with open(HIER_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("category", "") or ""
            if keyword.lower() in cat.lower():
                name = cat.rstrip("/").split("/")[-1].replace("_", " ")
                hits.append((row["landmark_id"], name, row.get("supercategory", "")))
    print(f"关键词 '{keyword}' 命中 {len(hits)} 个地标（展示前 40 个）：")
    for lid, name, sup in hits[:40]:
        print(f"  landmark_id={lid:>10}  {name}   [{sup}]")
    if not hits:
        print("未命中。换关键词试试，例如：Eiffel / Taj / Sydney / Forbidden / Pearl / Pisa / Great_Wall")


def step_fetch(landmark_ids, names, out_dir, per_class, workers) -> None:
    """从 train.csv 收集目标 landmark 的图片 URL 并下载。"""
    if not TRAIN_CSV.exists():
        raise SystemExit(f"缺少 {TRAIN_CSV}，请先执行 --step metadata")
    ids = [str(x).strip() for x in landmark_ids]
    if len(ids) != len(names):
        raise SystemExit(f"--landmark-ids（{len(ids)} 个）与 --names（{len(names)} 个）数量不一致")

    url_map = {lid: [] for lid in ids}
    print(f"扫描 {TRAIN_CSV}（约 501MB，通常需要 1~3 分钟）...")
    t0 = time.time()
    want = per_class * 3  # 多收集一些，下载失败/太小被剔除后仍够数
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_url, i_lid = header.index("url"), header.index("landmark_id")
        for row in reader:
            lid = row[i_lid]
            bucket = url_map.get(lid)
            if bucket is not None and len(bucket) < want:
                bucket.append(row[i_url])
        # 扫描完才知道每个 id 实际有多少张
    print(f"扫描完成，耗时 {time.time() - t0:.0f}s")
    for lid, name in zip(ids, names):
        print(f"  landmark_id={lid}: 可用 URL {len(url_map[lid])} 条")

    out_root = Path(out_dir)
    tasks = []
    counters = {}  # 同一类别名合并多个 landmark_id 时，文件名计数器跨 id 连续递增
    for lid, name in zip(ids, names):
        urls = url_map[lid]
        if not urls:
            print(f"[警告] landmark_id={lid} 在 train.csv 中没有图片，跳过")
            continue
        # 均匀抽样，避免前后几张雷同
        if len(urls) > per_class:
            step = len(urls) / per_class
            urls = [urls[int(i * step)] for i in range(per_class)]
        cls_dir = out_root / name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            tasks.append((url, cls_dir / f"{name}_{idx:03d}.jpg"))

    print(f"开始下载 {len(tasks)} 张图片（{workers} 线程）...")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_one, t): t for t in tasks}
        for fut in as_completed(futs):
            url, dst = futs[fut]
            try:
                success, msg = fut.result()
            except Exception as e:  # 兜底，单张失败不影响整体
                success, msg = False, f"{type(e).__name__}: {e}"
            if success:
                ok += 1
            else:
                fail += 1
                if dst.exists():
                    dst.unlink()
                if fail <= 5:
                    print(f"  [失败] {url[:80]} -> {msg}")
    print(f"下载完成：成功 {ok} 张，失败 {fail} 张")
    print("下一步：剔除明显无关/模糊/带水印的图，然后进入标注环节。")


def download_one(task):
    """下载单张图片并统一转存为 RGB JPEG；返回 (是否成功, 信息)。"""
    from PIL import Image

    url, dst = task
    if dst.exists() and dst.stat().st_size > 2048:
        return True, "cached"
    last_err = ""
    for _ in range(2):  # 失败重试一次
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) < 2048:
                return False, "文件过小（<2KB，多为占位图）"
            img = Image.open(io.BytesIO(data))
            img.load()
            img.convert("RGB").save(dst, "JPEG", quality=92)
            return True, ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1)
    return False, last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="GLDv2 定向下载工具")
    ap.add_argument("--step", required=True, choices=["metadata", "find", "fetch"])
    ap.add_argument("--name", default=None, help="find: 名称关键词")
    ap.add_argument("--landmark-ids", default=None, help="fetch: 逗号分隔的 landmark_id")
    ap.add_argument("--names", default=None, help="fetch: 与 landmark-ids 一一对应的类别目录名")
    ap.add_argument("--out", default="data/data_raw/gldv2")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.step == "metadata":
        step_metadata()
    elif args.step == "find":
        if not args.name:
            raise SystemExit("--step find 需要 --name 关键词")
        step_find(args.name)
    else:
        if not args.landmark_ids or not args.names:
            raise SystemExit("--step fetch 需要 --landmark-ids 和 --names")
        step_fetch(
            landmark_ids=args.landmark_ids.split(","),
            names=args.names.split(","),
            out_dir=args.out,
            per_class=args.per_class,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
