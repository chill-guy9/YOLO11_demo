"""
主数据源：必应图片爬虫采集（本机 GLDv2 路线因 Wikimedia 被墙不可用，改为主力）
依赖：pip install icrawler
注意：
  1) 仅用于学习/作业场景，注意图片版权，不要商用
  2) 必应页面结构可能变化，若爬不到结果，换 bing-image-downloader 或手动浏览器存图
  3) 爬完必须人工过一遍：剔除无关图、截图、水印严重的图（配合 auto_label.py 预标注后复核）
"""
from pathlib import Path

from icrawler.builtin import BingImageCrawler

# 类别名 -> 搜索关键词列表（中英文各一组，爬取后合并）
# 注意：与 data/classes.txt 的类别保持一致；新增类别时在这里补一行即可（多给几个同义词，
#       类内多样性越高模型越稳）
CLASSES = {
    "eiffel_tower": ["埃菲尔铁塔", "Eiffel Tower Paris"],
    "sydney_opera": ["悉尼歌剧院", "Sydney Opera House"],
    "oriental_pearl": ["东方明珠塔 上海", "Oriental Pearl Tower Shanghai"],
    "statue_of_liberty": ["自由女神像 纽约", "Statue of Liberty New York"],
    "great_wall": ["长城 慕田峪", "长城 司马台", "Great Wall of China Mutianyu"],
}

MAX_PER_QUERY = 40  # 每组关键词爬取数量（5 类 x 2~3 组 x 40 ≈ 400~600 张上限）
# icrawler 从 000001.jpg 开始编号，每个类别目录独立；注意新版默认输出到 data/data_raw，
# 旧版 README 里写的是 data/data_raw，两者是同一个用途，实际用哪个就以 data.yaml / 预标注脚本为准


def main() -> None:
    root = Path("data/data_raw")
    for name, keywords in CLASSES.items():
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for kw in keywords:
            print(f"[crawl] {name} <- {kw}")
            crawler = BingImageCrawler(
                storage={"root_dir": str(out_dir)},
                parser_threads=2,
                downloader_threads=4,
            )
            crawler.crawl(keyword=kw, max_num=MAX_PER_QUERY, file_idx_offset="auto")
        n = len([p for p in out_dir.iterdir() if p.is_file()])
        print(f"[done] {name}: {n} files -> {out_dir}")


if __name__ == "__main__":
    main()
