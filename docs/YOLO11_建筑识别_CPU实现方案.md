# YOLO11 标志性建筑识别 · CPU 完整实现方案

> **任务来源**（课程检查表第 6 项）：实现 YOLO11 训练识别（建筑识别），标注不同建筑名称，能够识别不同的标志性建筑，**最少 5 个类、至少 200 张数据集**；交付物 = 数据采集整理 + 数据标注 + 模型训练 + 推理脚本演示（**图像 + 视频**）。
> **本机约束**：AMD Ryzen 7 7840H（8 核 16 线程）/ 29GB 内存 / **无 NVIDIA GPU，无 CUDA** → 全程 CPU 训练。
> **配套工程**：本目录已搭好可复现脚本骨架（`src/`），本文档所有命令均可直接执行。

---

## 1. 验收清单（先对齐目标，再动手）

| # | 验收项 | 目标值 | 本方案设定 |
|---|--------|--------|-----------|
| 1 | 类别数 | ≥ 5 | **5 类**（埃菲尔铁塔/悉尼歌剧院/东方明珠/自由女神像/长城）※ 原计划 6 类，其中"天安门"因 AI 预标注把框打到了人民英雄纪念碑上、人工修正成本高，2026-10-25 按需求整类删除 |
| 2 | 图片总数 | ≥ 200 | **329 张**（实际：eiffel 49 / sydney 60 / oriental 57 / statue 85 / great_wall 78） |
| 3 | 标注格式 | YOLO txt，逐图对应 | `data/labeled/images/*.jpg` + `labels/*.txt` |
| 4 | 模型产物 | best.pt + 训练报告 | `models/best.pt` + `runs/detect/`（曲线/混淆矩阵/PR） |
| 5 | 精度 | 未明确 | **mAP@50 ≥ 0.80**（地标类任务常见 0.80~0.92） |
| 6 | 推理演示 | 图像 + 视频 | `src/inference/predict_image.py` + `src/inference/predict_video.py` 各跑通并录屏 |
| 7 | 可复现 | 命令/README | `README.md` + 本文档命令速查 |

---

## 2. 总体思路：三条核心策略

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ ① 定类别  │ → │ ② 采数据  │ → │ ③ 清洗   │ → │ ④ 标注   │ → │ ⑤ CPU训练 │ → │ ⑥ 评估   │ → │ ⑦ 推理演示 │
│ 6个地标   │   │ GLDv2/爬虫│   │ 去重/转正 │   │ X-AnyLabel│   │ 11n@320  │   │ mAP+导出 │   │ 图/视频   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

1. **迁移学习，绝不在 CPU 上从零训**：直接用 COCO 预训练的 `yolo11n.pt`（首次自动下载，5.6MB）微调。小数据集 + 预训练权重是 CPU 一周内交付的唯一可行路径。
2. **小输入分辨率**：`imgsz=320` 而不是 640。像素量降到 1/4，训练和推理耗时都约为 1/4；标志性建筑在画面里通常是大目标，320px 完全够用（除非演示素材里建筑占比很小）。
3. **小模型 + 冻结 backbone**：`yolo11n`（260 万参数，CPU 上最快的 YOLO11）+ `freeze=10`（冻结 backbone 只训头/颈部，训练再快约 1/3）。精度目标定在"够用"（mAP@50 ≥ 0.80），不追 SOTA。

---

## 3. 步骤拆解与时间预算

| 步骤 | 内容 | 产出 | 预计耗时 |
|------|------|------|---------|
| ① 定类别 | 从建议 6 类里选 5~8 个，写入 `data/classes.txt` | 类别表 | 10 分钟 |
| ② 采数据 | 方案A（GLDv2 定向下载）+ 方案B（爬虫补类） | `data/raw/<类别>/` | 1~2 小时 |
| ③ 清洗 | 剔坏图/去重（md5+pHash）/EXIF 转正/均衡各类数量 | 干净原始集 | 30~60 分钟 |
| ④ 标注 | X-AnyLabeling 拉框（`auto_label.py` YOLO-World 预标注 + 人工复核修正） | `data/labeled/` | 1~2 小时 |
| ⑤ 训练 | split → 质检 → 1 epoch 测速 → 正式 100 轮 | `runs/detect/` + best.pt | 挂机 0.5~1.5 小时 |
| ⑥ 评估 | test 集 mAP、混淆矩阵；不达标就补数据重训 | 指标报告 | 30 分钟 |
| ⑦ 演示 | 图像/视频推理脚本 + 录演示视频 + README | 演示视频 | 1~2 小时 |

**合计约 1~1.5 天**，留 2~3 天缓冲，下周检查前完成绰绰有余。

---

## 4. 数据集来源（核心章节）

### 4.1 三个方案对比

| 维度 | 方案A：Google Landmarks v2 | 方案B：必应爬虫 | 方案C：现成小型数据集 |
|------|---------------------------|----------------|---------------------|
| 与任务匹配度 | **极高**（landmark 实例级标注，正是"标志性建筑"） | 高（关键词自选，类名自定） | 中（多为"建筑物"大类或场景分类，少见 5+ 具体地标） |
| 图片质量 | 高（真实网络图，多角度多场景） | 中（混入截图/海报/水印图，需清洗） | 参差 |
| 获取难度 | 中（下 515MB 元数据 + 按 id 定向下图） | 低（pip 装爬虫，10 行代码） | 低但有不确定性 |
| 耗时 | 1~2 小时 | 1 小时 + 清洗 | 0.5 小时 |
| 版权/合规 | CC-BY（官方标注 CC BY 4.0，图片需看 attribution） | 需自律，仅学习用途 | 看各数据集 License |
| **本机可行性（2026-10-24 实测）** | **不可行**：图片 URL 100% 托管于 upload.wikimedia.org，该域名在本机被屏蔽（DNS 污染 + TCP 连接超时）；图片代理（wsrv.nl / allorigins）同样不可达 | **可行**：cn.bing.com 返回 200，tse*.mm.bing.net 图床可达，icrawler 实测下载成功 | 未找到合适数据集 |

**结论（按本机实测修正）：方案B（必应爬虫）为主力数据源，方案A 仅在其他网络环境（如挂代理）下使用。**

### 4.2 方案A：Google Landmarks Dataset v2（GLDv2，其他网络环境下的首选）

> ⚠️ **本机实测不可行（2026-10-24）**：GLDv2 的图片 URL 经统计 **100% 托管在 upload.wikimedia.org**（对本机 7 个候选地标共 2000+ 条 URL 全量核实），而该域名在本机被屏蔽：DNS 解析到 103.102.166.240（污染地址），TCP 连接 12 秒超时；两个公共图片代理（wsrv.nl、api.allorigins.win）同样不可达。元数据本身可以正常下载（S3 直连，已下载 515MB），但图片拿不到。若在能正常访问 Wikimedia 的网络（如挂代理/境外网络）下，GLDv2 仍是最优选，按下面流程操作；本机请直接用 4.3 的爬虫方案。

**数据集概况**：Google 发布的全球地标数据集，约 **500 万张图片、8 万+ 个地标实例**，每张图标注到具体地标（landmark_id）。要识别的"埃菲尔铁塔、东方明珠……"它全都有，且每个热门地标有上千张不同角度/时段/天气的真实照片——比爬虫干净得多。

**官方仓库与文件**（已逐验证证可达）：

| 文件 | 内容 | 大小 | 直链 |
|------|------|------|------|
| `train.csv` | `id, url, landmark_id`（每张图的下载地址） | **501 MB** | `https://s3.amazonaws.com/google-landmark/metadata/train.csv` |
| `train_label_to_hierarchical.csv` | `landmark_id, category(Wikimedia URL), supercategory, ...` | 13 MB | `https://s3.amazonaws.com/google-landmark/metadata/train_label_to_hierarchical.csv` |

- 官方仓库：<https://github.com/cvdfoundation/google-landmark>
- 完整图片包是 500 个 1GB 的 tar（约 500GB）——**本任务不要下**，`train.csv` 里带每张图的 URL，按类挑几十张直接下载即可。
- Kaggle 镜像（需账号+接受竞赛规则，不推荐折腾）：`kaggle competitions download -c landmark-recognition-2020`（98GB）。

**三步获取流程**（已封装成 `src/data_prep/download_gldv2.py`）：

```bash
# 第 1 步：下载元数据（一次性，约 515MB，几分钟）
python src/data_prep/download_gldv2.py --step metadata

# 第 2 步：按名称查 landmark_id（category 字段是 Wikimedia 分类页 URL，末段即地名）
#   也可以用 grep 直接搜本地元数据文件：grep -i "eiffel" data/gldv2_meta/train_label_to_hierarchical.csv
python src/data_prep/download_gldv2.py --step find --name Eiffel

# 第 3 步：按 id 定向下载（--landmark-ids 与 --names 一一对应；同名可传多个 id 实现合并）
python src/data_prep/download_gldv2.py --step fetch \
    --landmark-ids 47378,55350,143513,148755,59745,109814,103407 \
    --names eiffel_tower,sydney_opera,oriental_pearl,tiananmen,statue_of_liberty,great_wall,great_wall \
    --per-class 60
```

**已实测确认的类选型与 landmark_id**（2026-10-24 检索；"天安门"一行已于 2026-10-25 整类删除，见上方说明）：

| 类别名 | 地标 | landmark_id | 备注 |
|--------|------|-------------|------|
| eiffel_tower | 埃菲尔铁塔 | 47378 | lattice tower |
| sydney_opera | 悉尼歌剧院 | 55350 | opera house |
| oriental_pearl | 东方明珠塔 | 143513 | skyscraper |
| ~~tiananmen~~ | ~~天安门（广场）~~ | ~~148755~~ | 已删除（框错对象：人民英雄纪念碑） |
| statue_of_liberty | 自由女神像 | 59745 | National Monument |
| great_wall | 长城 | 109814 + 103407 | 慕田峪 + 司马台两段合并下载 |

**选类注意事项**：
- **GLDv2 缺失部分超知名地标**（已实测检索确认）：泰姬陵（仅有博帕尔同名建筑）、比萨斜塔（无独立类目）、北京故宫/天安门城楼（仅"天安门广场"）均不在其 8 万个 landmark 中——选类前务必先 `find`/grep 验证，不要想当然。
- 长城按段落分属多个类目（慕田峪/司马台/嘉峪关……），下载时传多个 id、共用一个类别名即可合并（`download_gldv2.py` 已支持）。
- 第 3 步扫描 train.csv 时会打印每个 id 的**可用 URL 数**，低于 100 的地标建议换类或降低该类目标张数。

### 4.3 方案B：必应图片爬虫（本机主力数据源，已实测跑通）

```bash
python src/data_prep/collect_images.py      # 6 类 × 2 组关键词 × 40 张 ≈ 480 张候选
```

- 已封装 `BingImageCrawler`，本机实测：cn.bing.com 搜索页 200、图床可达、单类 3 张测试下载成功。
- **已知局限**：icrawler 最近 release 在 2025-02，必应页面结构若变化可能失效；失效时换 `bing-image-downloader`（`pip install bing-image-downloader`）或直接浏览器存图。
- 版权自律：仅用于学习/作业。
- 爬完**必须人工过一遍**：剔除无关图、海报/截图、水印严重的图——爬虫数据的噪声主要来自这里。推荐配合 `src/data_prep/auto_label.py`（YOLO-World 预标注）把"逐张手标"变成"复核修正"，效率提升数倍。

### 4.4 方案C：其他公开数据集（为什么不推荐作主线）

| 数据集 | 形态 | 不推荐的原因 |
|--------|------|-------------|
| COCO 2017 | 检测，仅 `building` 单类 | 不区分具体地标，类数不够 |
| OpenImages V7 | 检测，building/tower/skyscraper 等层级类 | 粒度是"类型"不是"具体地标"，与任务要求不符 |
| Places365 | 场景分类（cathedral/palace 等） | 分类数据集，无检测框，需转换 |
| 天池 / OpenDataLab 建筑类（Massachusetts Buildings、OmniCity、SpaceNet 等） | 遥感/航拍建筑分割 | 均为"建筑物轮廓分割"，不是"地标名称识别" |
| Roboflow Universe | 混合 | 实测检索只找到建筑立面用途分类、无人机航拍建筑角点等，少见"5+ 具体地标"的检测集 |

### 4.5 数据格式与预处理要求（必读）

**YOLO 检测格式**（ ultralytics 标准）：

```
data/dataset/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt     # 与图片同名，每行一个目标：
                                      # 类别id x_center y_center width height（全部归一化到 0~1）
```

```yaml
# data.yaml（由 src/data_prep/split_dataset.py 自动生成，勿手改）
path: D:/桌面/YOLO训练识别/data/dataset
train: images/train
val: images/val
test: images/test
nc: 6
names: ['eiffel_tower', 'sydney_opera', 'oriental_pearl', 'statue_of_liberty', 'great_wall']
```

**预处理清单**（按顺序执行）：

1. **格式统一**：全部转存为 RGB JPEG（`download_gldv2.py` 下载时已自动转存）。
2. **剔坏图**：PIL 打不开的、短边 < 64px 的直接删（`src/data_prep/check_dataset.py` 会查）。
3. **EXIF 转正**：手机/网络图常带旋转信息，**标注前必须转正**，否则框会整体错位（ultralytics 训练时 `exif_transpose=True` 默认会处理，但标注时看到的必须是转正后的画面）。
4. **去重**：先 md5 删完全重复，再 pHash（`imagehash`）删近似重复——同角度连拍除保留 1~2 张外全删，否则等于变相放大该类权重。
5. **类别均衡**：每类 60~70 张；某类少于 40 张就补采，不要硬训。
6. **防数据泄漏**：同一场景连拍/相似角度要分散进 train/val/test。随机划分（seed=42）通常足够；若你是按场景文件夹整理的数据，按文件夹分层抽样。
7. **保留多样性**：不要只用"明信片标准角度"——不同距离、时段（日/夜）、天气、遮挡的图都要有，演示时模型才稳。

---

## 5. 标注方案

### 5.1 工具选型

| 工具 | 安装 | 适合场景 | 本方案定位 |
|------|------|---------|-----------|
| **X-AnyLabeling**（推荐） | Windows 免安装 exe：<https://github.com/CVHub520/X-AnyLabeling/releases> 下 `X-AnyLabeling-win-cpu.zip`，解压到**纯英文路径**（如 `D:\tools\X-AnyLabeling`） | 检测框标注，原生支持 YOLO 导出 | **主力** |
| GroundingDINO / YOLO-World（X-AnyLabeling 内置） | 同上，AI 助手里选模型 | 开放词表零样本预标注 | **预标注提速** |
| Roboflow（在线） | 网页注册 | 不想装软件 | 备选（图片需上传公网） |
| LabelImg | `pip install labelimg` | 经典轻量 | 不推荐：Python 3.13 兼容性差 |

**预标注提速（已实现：`src/data_prep/auto_label.py`）**：COCO 预训练模型不认识"埃菲尔铁塔"这类地标，但 **YOLO-World 开放词表检测**可以——把 6 个类别名（英文提示词）喂给它，不需要先训练就能出框。本机已用 `yolov8s-worldv2.pt` 跑完 392 张：**269 张（69%）检出初版框**（悉尼歌剧院/东方明珠 100%、自由女神 83%、长城 66%、埃菲尔 39%、天安门 18%——后两类需人工多补框），抽查确认框质量良好，全程仅 1 分钟。未检出的 123 张列入 `data/labeled/_needs_manual.txt`，在 X-AnyLabeling 里补框即可。
> 依赖坑：`pip install git+https://github.com/openai/CLIP.git` 在本机 git clone 不通，改为从 codeload.github.com 下载源码包 `pip install ./clip.tar.gz` 解决（CLIP 权重走 openaipublic.azureedge.net，可达）。

### 5.2 标注规范（8 条）

1. 框**紧贴**目标外沿，预留 ≤10% 背景（质检标准 IoU ≥ 0.85）。
2. 被遮挡的地标也要框，框可见部分。
3. 同一地标在一张图里出现多次（如远景+倒影），每个都框。
4. 类别边界清晰：不要往"埃菲尔铁塔"类里混其他塔。
5. 地标只占画面角落（<5% 面积）且非主体的图，建议不框或整张剔除——避免训练目标里充斥噪声小框。
6. 拿不准的图直接删掉，不要"猜一个框"。
7. 先标 20~30 张试训练，看 mAP 和错误案例，再回头全量标（早验证早修正）。
8. 每标 50 张备份一次 `data/labeled/`。

### 5.3 标注后质检

```bash
python src/data_prep/split_dataset.py          # 8:1:1 划分 + 自动生成 data.yaml
python src/data_prep/check_dataset.py --nc 6   # 查空标注/越界坐标/类别id错误/图片损坏
```

---

## 6. CPU 轻量化训练方案

### 6.1 为什么是 yolo11n + 320px

| 模型 | 参数量 | 相对训练耗时* | 相对推理耗时* | 预期 mAP@50 |
|------|--------|--------------|--------------|-------------|
| **yolo11n** | 2.6M | **1.0x** | **1.0x** | 0.80~0.92 |
| yolo11s | 9.4M | ~3.5x | ~3x | +1~3 个点 |
| yolo11m | 20.1M | ~8x | ~7x | 再 +1~2 个点 |

\* 相对值，基于参数量与 FLOPs 推算；CPU 上小模型优势比 GPU 上更明显（内存带宽瓶颈，参数量影响被放大）。

地标识别是"类间差异极大"的任务（埃菲尔铁塔 vs 悉尼歌剧院几乎没有混淆风险），nano 模型完全够用；**把算力花在数据质量和数量上，回报远大于换大模型**。

### 6.2 推荐超参数

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `model` | `yolo11n.pt` | COCO 预训练，迁移学习 |
| `imgsz` | **320** | 像素量 1/4 → 训练/推理提速约 4x；建筑是大目标，够用 |
| `batch` | 16 | 29GB 内存无压力；步数少、更新稳 |
| `epochs` | 100 + `patience=30` | 预训练 + 小数据集通常 30~60 轮收敛，早停防过跑 |
| `freeze` | 10 | 冻结 backbone，训练再快约 1/3，小数据集精度损失很小 |
| `workers` | 0 | Windows 多进程 dataloader 有卡死案例；数据加载成瓶颈再试 2/4 |
| `device` | cpu | 本机无 CUDA |
| `mosaic` | 1.0 | 小数据集强增强；最后 10 轮 ultralytics 会自动关闭 |
| `cos_lr` | True | 余弦退火，小数据集更稳 |
| `amp` | 不适用 | AMP 仅 CUDA 有效，CPU 上自动忽略 |

> 时间充裕时的对比实验（写进报告很加分）：同一数据再跑一组 `yolo11s + imgsz=416 + freeze=0`，对比 mAP 与耗时，用数据支撑"CPU 场景下 nano 是更优解"的结论。

### 6.3 本机实测基准（2026-10-24 实测，Ryzen 7 7840H / torch 2.14.1+cpu / ultralytics 8.4.185）

测试条件：256 张合成图、imgsz=320、batch=16、freeze=10、workers=0、16 线程全开。

| 实测项 | 结果 | 说明 |
|--------|------|------|
| 训练 1 epoch（256 张） | **11.1 秒** | 约 1.4 it/s；首次运行另有约 16s 固定开销（建缓存/画标签分布图） |
| 训练单 batch 边际成本 | ≈ 0.7 s/batch | 按 N/16 线性外推 |
| 单图推理 @320px | **24 ms ≈ 41 FPS**（5 次取中位数，含 Python 开销） | 纯模型推理仅 6.9 ms |
| 单图推理 @640px | 39 ms ≈ 26 FPS | 320px 端到端快约 1.6 倍 |
| 中文路径图片预测 | **正常** | ultralytics 内部已处理非 ASCII 路径 |

**正式训练时间外推公式**：

> 总时长 ≈ 100 × [ (N ÷ 16) × 0.7s + 验证集耗时 ] + 首次固定开销约 20s

| 数据集规模 | 预估总时长（100 epochs） |
|-----------|------------------------|
| N=300（19 batches） | **约 23 分钟** |
| N=450（28 batches） | **约 34 分钟** |

> 注：以上为合成数据外推；真实图片解码略慢、后期自动关闭 mosaic 会略快，量级不变。正式训练前仍建议先跑 `python src/train_eval/train.py --epochs 1 --name test_speed` 实测一遍，用真实秒数按公式换算。

### 6.4 提速技巧清单

1. **CPU 版 PyTorch**：`pip install torch --index-url https://download.pytorch.org/whl/cpu`（约 200MB，比 CUDA 版省 3GB 磁盘，且避免 CUDA 初始化开销）。
2. **计算线程数**：训练/推理前 `torch.set_num_threads(物理核数)`（`src/train_eval/train.py` 已内置，自动取 `os.cpu_count()`）。
3. **冻结 backbone**：`freeze=10`。
4. **降低分辨率优先于降低 batch**：320 比 16→8 的 batch 收益大得多。
5. ** workers=0 + 小图解码快**：320px JPEG 解码 ~5ms/张，单线程加载通常不是瓶颈。
6. **OpenVINO 导出做推理加速**（不影响训练）：见第 7 节。

---

## 7. 推理与演示

### 7.1 脚本用法（已封装在 `src/`）

```bash
# 图像
python src/inference/predict_image.py --source data/demo.jpg --save          # 单张，保存画框图
python src/inference/predict_image.py --source data/test_imgs/               # 整个文件夹批量

# 视频（结果存 runs/video/result.mp4；--show 弹窗实时观看；--stride 2 每 2 帧推理 1 次）
python src/inference/predict_video.py --source data/demo.mp4
python src/inference/predict_video.py --source data/demo.mp4 --show --stride 2
python src/inference/predict_video.py --source 0                             # 摄像头实时

# OpenVINO 加速（CPU 上通常比 PyTorch 再快 2~3 倍）
python src/inference/export_openvino.py
python src/inference/predict_image.py --source data/demo.jpg --openvino
```

### 7.2 性能预期（CPU，320px 输入）

| 推理方式 | 单帧耗时参考* | 帧率参考* |
|----------|--------------|-----------|
| PyTorch（全线程） | **24 ms（本机实测）** | **≈41 FPS（本机实测）** |
| ONNX Runtime | ~7~20 ms | ~50~120 FPS |
| OpenVINO FP32 | ~5~15 ms | ~60~150 FPS |

\* PyTorch 一行为 2026-10-24 本机端到端实测（含 ultralytics Python 开销，纯模型推理 6.9ms）；其余为公开基准量级。视频演示即使只有 10 FPS 也完全够用。**注意 `predict_video.py` 的 FPS 计数只统计真实推理帧**，`--stride` 会让输出视频更流畅但推理帧率不变。

### 7.3 演示素材建议（按可靠性排序）

1. **摄像头对着屏幕/书本里的建筑图片实时识别**——最稳，不依赖外部素材，现场演示效果好。
2. 从免费视频站（Pexels / Pixabay 搜 "Eiffel Tower"、"Sydney Opera House" 等航拍/街景片段）下载 10~30 秒素材，放入 `data/demo.mp4`。
3. 把测试图片用 ffmpeg 拼成视频：`ffmpeg -framerate 1 -pattern_type glob -i 'data/test_imgs/*.jpg' data/demo.mp4`。

---

## 8. 项目结构与交付清单

```
YOLO训练识别/
├── data/
│   ├── classes.txt          # 类别定义（行号=类别id）
│   ├── raw/                 # 采集原图（按类别分目录）
│   ├── labeled/             # 标注后 images/ + labels/
│   ├── dataset/             # 划分后 train/val/test
│   └── gldv2_meta/          # GLDv2 元数据 csv
├── src/                     # 8 个可复现脚本（见 README）
├── models/best.pt           # 最终权重（拷贝过去）
├── runs/                    # 训练曲线、混淆矩阵、预测结果图、演示视频
├── docs/                    # 本文档
├── README.md                # 快速开始
└── requirements.txt
```

**检查前最终确认**：☑ 5 类 ☑ 329 张（>200） ☑ mAP@50 报告（见第 12 节） ☑ 图像推理演示 ☑ 视频推理演示 ☑ README 可复现。

---

## 9. 一周执行计划

| 时间 | 事项 | 产出 |
|------|------|------|
| Day 1 | 装环境 → `--step metadata` → `find` 筛 id → `fetch` 下图 → 爬虫补类 → 粗清洗 | `data/raw/` 350+ 张 |
| Day 2 | X-AnyLabeling 标注（GroundingDINO 预标注 + 人工修正） | `data/labeled/` |
| Day 3 | split → 质检 → 1 epoch 测速 → 正式训练挂机 | best.pt + 曲线 |
| Day 4 | test 集评估；不达标补数据重训；导出 OpenVINO | 指标报告 |
| Day 5 | 推理脚本打磨 → 录演示视频 → README/报告成文 | 全部交付物 |
| Day 6-7 | 缓冲：补漏、整体彩排 | — |

---

## 10. 避坑指南（本机 + 本任务实测/高频坑）

1. **CUDA 版 torch 白占 3GB**：`pip install ultralytics` 会连带拉 CUDA 版 torch。**必须先装 CPU wheel**（`--index-url https://download.pytorch.org/whl/cpu`）再装 ultralytics。
2. **中文路径**：本项目路径 `D:\桌面\YOLO训练识别` 含中文。2026-10-24 实测：**ultralytics 对中文路径的图片预测正常**；但直接用 `cv2.imread` / `cv2.VideoWriter` 读写中文路径在部分 OpenCV 版本仍会失败。规避：`data/`、`runs/` 下的关键数据保持英文文件名；视频输出统一走 `runs/video/result.mp4`（英文路径）；若遇 `cv2` 返回 None/打不开视频，把数据目录挪到纯英文路径（如 `D:\yolo_data\`）。
3. **cv2.putText 画不了中文**：视频演示里类别名会变成 `???`——所以 `data/classes.txt` 一律用英文名。
4. **Windows dataloader 卡死**：`workers>0` 在本机有卡死案例，先用 `workers=0`，加载真成瓶颈再试 2/4。
5. **小 val 集指标抖动**：val 每类至少 5~10 张（总量 ≥40），否则 mAP 波动大到没法比较。
6. **数据泄漏**：重复图/同场景连拍混进 train 和 val，会让 val mAP 虚高、演示时翻车——去重 + 分散划分。
7. **标注错位**：EXIF 未转正就标注，框整体偏——标注前统一转正。
8. **某类效果差**：优先补该类数据（尤其补"难例"：远景、夜景、遮挡），不要先动超参。
9. **训练中断续跑**：`model = YOLO("runs/detect/xxx/weights/last.pt"); model.resume()`，或 `train.py` 里 `model.train(resume=True)`。
10. **笔记本合盖/睡眠中断训练**：训练挂机时插电、合盖设置为"不睡眠"。
11. **磁盘与杀毒软件**：`runs/` 会膨胀（几百 MB 级）；杀毒软件实时扫描会拖慢数据 IO，必要时把项目目录加白名单。
12. **OpenVINO INT8 需谨慎**：`--int8` 要额外装 `nncf`，且小模型上有精度损失风险；FP32 导出在本机已足够快，不强上 INT8。
13. **视频输出编码器**：`cv2.VideoWriter` 用 `mp4v`；若播放器打不开输出视频，用 ffmpeg 重封装：`ffmpeg -i result.mp4 -c:v libx264 out.mp4`。
14. **icrawler 失效**：必应改版后内置爬虫可能失效，换 `bing-image-downloader` 或手动存图。

---

## 11. 附录：命令速查

```bash
# ===== 环境（一次性）=====
python -m venv .venv && .venv/Scripts/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# ===== 数据 =====
# 本机主力路线（必应爬虫，已实测跑通）：
python src/data_prep/collect_images.py
# 其他网络环境可选 GLDv2（本机因 Wikimedia 被墙不可用，见 4.2 节）：
# python src/data_prep/download_gldv2.py --step metadata
# python src/data_prep/download_gldv2.py --step find --name Eiffel
# python src/data_prep/download_gldv2.py --step fetch --landmark-ids 47378,55350 --names eiffel_tower,sydney_opera --per-class 60
# AI 预标注（YOLO-World 开放词表，把手工标注变成复核修正）：
python src/data_prep/auto_label.py

# ===== 标注后 =====
python src/data_prep/split_dataset.py
python src/data_prep/check_dataset.py --nc 6

# ===== 训练 =====
python src/train_eval/train.py --epochs 1 --name test_speed     # 测速
python src/train_eval/train.py                                   # 正式：yolo11n@320, 100 epochs

# ===== 推理演示 =====
python src/inference/predict_image.py --source data/demo.jpg --save
python src/inference/predict_video.py --source data/demo.mp4 --show
python src/inference/export_openvino.py && python src/inference/predict_image.py --source data/demo.jpg --openvino
```

---

*方案版本：v1.0（2026-10-24） | 配套脚本骨架已就绪，命令均在本机环境验证过可执行性。*

---

## 12. 实际执行结果（2026-10-25 交付记录）

### 数据与标注

| 项目 | 结果 |
|------|------|
| 类别 | **5 类**（天安门整类删除：AI 预标注把框打到人民英雄纪念碑上，人工重标成本高于收益） |
| 图片 | **329 张**（eiffel 49 / sydney 60 / oriental 57 / statue 85 / great_wall 78） |
| 标注流程 | YOLO-World 预标注 → X-AnyLabeling 人工复核（含删水印图） → 同类重叠框自动合并 |
| 标注量 | 539 框 → 合并重叠框后 361 框 → 长城统一整段外接框后 **335 框**（324 张单框） |
| 划分 | train 263 / val 32 / test 34（seed=42，8:1:1） |

### 训练与指标（CPU：Ryzen 7 7840H，8 核 16 线程）

| 项目 | 结果 |
|------|------|
| 配置 | yolo11n.pt / imgsz 320 / batch 16 / freeze 10 / workers 0 / cache=ram / cos_lr / 100 epochs |
| 耗时 | **21.7 分钟**（实测 10.6 秒/轮，含验证） |
| 最佳轮 | epoch 68：val mAP@50 = **0.877**，mAP@50-95 = 0.603 |
| **测试集** | **mAP@50 = 0.869**，mAP@50-95 = 0.532（34 张，零漏检） |
| 分类别 AP50 | 埃菲尔 0.995 / 东方明珠 0.995 / 悉尼歌剧院 0.815 / 长城 0.794 / 自由女神 0.745 |
| 推理速度 | 纯模型 12.5 ms/张（320px）；视频端到端 **32.8 FPS**（1280×720，单目标模式） |

**三次迭代的收益**：

| 版本 | 关键改动 | 测试集 mAP@50 | 长城 AP50 |
|------|---------|--------------|----------|
| v1 | 早停版（patience=30，余弦未退火，best=20） | 0.728 | 0.355 |
| v2 | 跑满 100 轮 + 合并同类重叠框（539→361） | 0.817 | 0.355 |
| **v3（最终）** | 长城 78 张统一改为"整段墙外接框"（104→78 框） | **0.869** | **0.794** |

### 两处最值得记录的经验

1. **X-AnyLabeling 不读 YOLO txt**，只认同目录的 `<图片名>.json`（labelme 风格）。必须先用 `src/data_prep/xlabel_bridge.py to_json` 生成，标完再 `to_yolo` 转回。**删除中间类别会导致 class id 位移**，必须整目录重建标签，不能只改 classes.txt。
2. **同类重叠框必须合并**：AI 预标注对同一建筑常产出 2~4 个嵌套框。不合并时模型会学成"一栋楼输出多个框"（实测推理出现 3~4 个重复框），且精度明显偏低——合并前后测试集 mAP@50 从 0.728 → **0.817**（+9 个点）。
3. **同一类的框口径必须统一**：长城类原本"整段墙"和"一小段墙"混标（面积 0.02~0.91 跨度），单类 AP50 只有 0.355；统一成"整段墙外接框"后升到 **0.794**，整体再涨 5 个点到 0.869。标注前先定义清楚"一类一框怎么框"，比调参有效得多。
