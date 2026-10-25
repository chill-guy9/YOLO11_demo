# YOLO11 标志性建筑识别（CPU 训练版）

针对**纯 CPU 机器**（本机：AMD Ryzen 7 7840H / 8核16线程 / 29GB 内存 / 无 NVIDIA GPU）的
YOLO11 建筑识别完整工程。任务要求：≥5 类标志性建筑、≥200 张图片、数据采集+标注+训练+推理演示（图像+视频）。

> 📄 **完整实现方案（含数据集来源、CPU 训练权衡、避坑指南）见 `docs/YOLO11_建筑识别_CPU实现方案.md`**
> 📘 **从采集到训练到测试的完整流程手册见 `docs/完整流程与训练步骤.md`**（标注完成后怎么训、怎么评估，看这份）
> 📊 **训练结果与评估报告见 `docs/训练评估报告.md`**（双集指标、长城短板归因、320/480 消融）

## 目录结构

```
├── data/
│   ├── classes.txt              # 类别定义（行号=类别 id，共 5 类）
│   ├── dataset/                 # 训练数据：images/ + labels/（train 263 / val 32 / test 34）
│   ├── labeled/                 # 标注源：329 张图 + X-AnyLabeling json + YOLO txt
│   ├── data_raw/                # 采集的原始图（330 张，按类别分文件夹）
│   ├── demo_imgs/               # 图像推理演示输入（10 张）
│   └── demo.mp4                 # 视频推理演示输入（58.8s，1280×720）
├── src/
│   ├── data_prep/               # ① 数据准备：采集 → 预标注 → 清洗 → 划分 → 质检
│   │   ├── collect_images.py    #   必应爬虫采集
│   │   ├── download_gldv2.py    #   GLDv2 定向下载（本机 Wikimedia 被墙，不可用）
│   │   ├── auto_label.py        #   YOLO-World 开放词表自动预标注
│   │   ├── xlabel_bridge.py     #   YOLO txt ↔ X-AnyLabeling json 互转
│   │   ├── merge_boxes.py       #   合并同类重叠框（一图一框）
│   │   ├── split_dataset.py     #   8:1:1 划分 + 生成 data.yaml
│   │   ├── check_dataset.py     #   基础质检（配对/越界/空标注）
│   │   ├── qc_labels.py         #   深度质检（类别一致性/重复框/极小框）
│   │   └── supplement.py        #   定向补数据（爬取→去重→预标注→筛选）
│   ├── train_eval/              # ② 训练与评估
│   │   ├── train.py             #   CPU 训练入口（yolo11n@320, 100 epochs ≈ 22 分钟）
│   │   ├── eval_compare.py      #   多权重 × 多划分对比评估（输出 Markdown 表）
│   │   └── benchmark.py         #   PyTorch vs OpenVINO 端到端/前向基准
│   ├── inference/               # ③ 推理与演示
│   │   ├── predict_image.py     #   图像推理
│   │   ├── predict_video.py     #   视频/摄像头推理（可 --stride 跳帧）
│   │   ├── make_demo_video.py   #   用图片合成带运镜的演示视频（无需联网）
│   │   └── export_openvino.py   #   导出 OpenVINO 模型
│   └── tools/                   # ④ 工具
│       └── recycle_tool.py      #   安全删除（走系统回收站，可恢复）
├── models/
│   ├── best.pt                  # 最终模型（320px, test mAP50 0.869）
│   ├── best_480.pt              # 480px 高召回备选（mAP50 0.919）
│   ├── best_openvino_model/     # OpenVINO IR
│   └── archive_best_*.pt        # 旧标注训练的对照权重
├── runs/
│   ├── detect/<实验名>/         # 训练曲线、混淆矩阵、PR/F1、weights/
│   ├── video/result.mp4         # 视频推理成品
│   ├── demo_predict/            # 图像推理结果
│   └── demo_grid.jpg            # 结果九宫格
├── docs/                        # 方案、评估报告、清理记录
├── requirements.txt
└── data.yaml                    # 由 split_dataset.py 自动生成
```

## 快速开始

> 所有命令请在**项目根目录**（本 README 所在目录）下执行，脚本内的相对路径都基于根目录。

### 0. 环境（一次性）

```bash
python -m venv .venv && .venv/Scripts/activate
# CPU 版 torch 必须先单独装，否则会拉 3GB 的 CUDA 版
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### 1. 数据采集（本机主力：必应爬虫）

> ⚠️ 本机实测：GLDv2 图片 100% 托管于 upload.wikimedia.org，该域名被屏蔽（DNS 污染+超时），
> 图片代理也不通——**GLDv2 直连路线在本机不可用**（详见方案文档 4.2 节），故以必应爬虫为主力。

```bash
python src/data_prep/collect_images.py      # 6 类 × 2 组关键词 × 40 张 ≈ 480 张候选
```

其他网络环境（可访问 Wikimedia）下可选 GLDv2 公开数据集，已实测选定的 6 类及 landmark_id：

```bash
python src/data_prep/download_gldv2.py --step fetch \
    --landmark-ids 47378,55350,143513,148755,59745,109814,103407 \
    --names eiffel_tower,sydney_opera,oriental_pearl,tiananmen,statue_of_liberty,great_wall,great_wall \
    --per-class 60
```

### 2. 标注

用 **X-AnyLabeling**（Windows 免安装 exe：<https://github.com/CVHub520/X-AnyLabeling/releases>，
下 `X-AnyLabeling-win-cpu.zip`，解压到**纯英文路径**）逐张拉框，导出 YOLO 格式 txt；
把图片和 txt 分别放进 `data/labeled/images/` 和 `data/labeled/labels/`。

### 3. 划分与质检

```bash
python src/data_prep/split_dataset.py        # 8:1:1 划分 + 生成 data.yaml
python src/data_prep/check_dataset.py --nc 6 # 质检（类别数按实际改）
```

### 4. 训练（CPU）

```bash
python src/train_eval/train.py                                  # 默认 yolo11n@320, 100 epochs
python src/train_eval/train.py --epochs 1 --name test_speed     # 先跑 1 epoch 测速，看 s/epoch 再决定总轮数
```

### 5. 推理演示

```bash
# 图像
python src/inference/predict_image.py --source data/demo.jpg --save
python src/inference/predict_image.py --source data/test_imgs/

# 视频（结果存 runs/video/result.mp4；--show 弹窗观看）
python src/inference/predict_video.py --source data/demo.mp4
python src/inference/predict_video.py --source data/demo.mp4 --show --stride 2

# 可选：OpenVINO 加速（CPU 上通常再快 2~3 倍）
python src/inference/export_openvino.py
python src/inference/predict_image.py --source data/demo.jpg --openvino
```

## 项目状态：已完成 ✅（2026-10-25）

| 环节 | 结果 |
|------|------|
| 类别 | **5 类**：eiffel_tower / sydney_opera / oriental_pearl / statue_of_liberty / great_wall（天安门因预标注框错对象，按需求整类删除） |
| 数据 | **329 张**（eiffel 49 / sydney 60 / oriental 57 / statue 85 / great_wall 78），必应爬虫采集 + md5/pHash 去重 |
| 标注 | AI 预标注（YOLO-World）→ X-AnyLabeling 人工复核 → 同目标重叠框合并（539→361 框）→ 长城类统一为整段墙外接框（104→78 框） |
| 训练 | yolo11n @320px / batch16 / freeze10 / **100 epochs 跑满 21.7 分钟**（CPU） |
| **验证集** | 独立复评（`src/train_eval/eval_compare.py`）：**mAP@50 = 0.872**，mAP@50-95 = 0.584〔训练日志内 val 峰值为 0.877 / 0.603，差异来自训练内验证的批处理拼接方式，属正常波动〕 |
| **测试集** | **mAP@50 = 0.869**，mAP@50-95 = 0.532（34 张，零漏检） |
| 分类别 AP50 | 埃菲尔 0.995 / 东方明珠 0.995 / 悉尼歌剧院 0.815 / 长城 0.794 / 自由女神 0.745〔自由女神 test 仅 4 张，val 为 0.944〕 |
| 推理 | 图像 10 张全对，置信度 **0.50~0.91**；视频 1469 帧 @1280×720，**28.5 FPS** |
| 产物 | `models/best.pt`（5.5MB）、`runs/detect/landmark5_v3_wallfix/`（曲线/混淆矩阵）、`runs/video/result.mp4`（演示视频）、`models/best_480.pt`（480px 高召回备选） |

> 两次修正带来的提升：早停版 0.728 → 合并重叠框 0.817 → 长城标注统一 **0.869**（长城单类 0.355 → 0.794）。

> 标注格式换坑记录：X-AnyLabeling 打开目录只认自己的 `.json`，不读 YOLO `.txt`——
> 用 `python src/data_prep/xlabel_bridge.py to_json` 转过去，标完用 `to_yolo` 转回来（已封装，双向验证 512 框守恒）。

### 复现命令

```bash
python src/data_prep/xlabel_bridge.py to_yolo        # JSON -> YOLO（若改了标注）
python src/data_prep/split_dataset.py                # 8:1:1 划分 + 生成 data.yaml
python src/data_prep/check_dataset.py --nc 5         # 质检
python src/train_eval/train.py --epochs 100 --patience 100 --cache ram   # 训练（约 20 分钟）
python src/inference/predict_image.py --source data/dataset/images/test --save --max-det 1
python src/inference/predict_video.py --source D:/yolo_demo/demo.mp4 --max-det 1
```

| 实测项 | 结果 |
|--------|------|
| 训练 1 epoch（256 张, 320px, batch 16, freeze 10） | **11.1 秒**（约 1.4 it/s） |
| 单图推理 @320px | **24 ms ≈ 41 FPS**（端到端；纯模型推理 6.9 ms） |
| 单图推理 @640px | 39 ms ≈ 26 FPS |
| 中文路径图片预测 | 正常 |

结论：

- 训练主线：`yolo11n + imgsz=320 + batch=16 + freeze=10 + 100 epochs`
- 数据集 300~450 张时，正式训练约 **23~34 分钟**（外推公式见方案文档 6.3 节）
- 推理（320px）：PyTorch 端到端约 36~41 FPS，视频演示绰绰有余

> **实测修正（2026-10-25，`src/train_eval/benchmark.py`）**：OpenVINO 在本任务上**没有** 2~3 倍提速。
> 单图端到端 PyTorch 27.3ms vs OpenVINO 27.1ms（1.01x），纯前向 19.4ms vs 17.7ms（1.09x）。
> 原因是模型仅 2.6M 参数、输入 320px，端到端瓶颈在 Python 前后处理而非卷积算子。
> OpenVINO 只在 imgsz≥640、更大模型或 batch>1 时才值得用。
>
> 完整双集评估、长城短板归因、两版模型对比见 **`docs/训练评估报告.md`**。
