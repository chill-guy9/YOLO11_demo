# YOLO11 标志性建筑识别（CPU 训练版）

针对**纯 CPU 机器**（本机：AMD Ryzen 7 7840H / 8核16线程 / 29GB 内存 / 无 NVIDIA GPU）的
YOLO11 建筑识别完整工程。任务要求：≥5 类标志性建筑、≥200 张图片、数据采集+标注+训练+推理演示（图像+视频）。

> 📄 **完整实现方案（含数据集来源、CPU 训练权衡、避坑指南）见 `docs/YOLO11_建筑识别_CPU实现方案.md`**

## 目录结构

```
├── data/
│   ├── classes.txt              # 类别定义（行号=类别id，建议英文名）
│   ├── raw/                     # 采集的原始图片（按类别分文件夹）
│   ├── labeled/                 # 标注后：images/ + labels/（平坦式）
│   ├── dataset/                 # 划分后的 train/val/test（由脚本生成）
│   └── gldv2_meta/              # GLDv2 元数据 csv（脚本下载）
├── src/
│   ├── download_gldv2.py        # 方案A：GLDv2 公开数据集定向下载
│   ├── collect_images.py        # 方案B：必应图片爬虫采集（备选）
│   ├── split_dataset.py         # 划分 8:1:1 并生成 data.yaml
│   ├── check_dataset.py         # 训练前质检
│   ├── train.py                 # CPU 训练入口
│   ├── predict_image.py         # 图像推理
│   ├── predict_video.py         # 视频/摄像头推理
│   └── export_openvino.py       # OpenVINO 加速导出（可选）
├── models/                      # 存放 best.pt / OpenVINO 模型
├── runs/                        # 训练与推理输出
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

## 当前进度（2026-10-24）

- [x] **环境**：torch 2.14.1+cpu / ultralytics 8.4.185 / opencv / icrawler / CLIP 已装好
- [x] **数据采集**：必应爬虫 392 张（eiffel 49 / sydney 60 / oriental 57 / tiananmen 62 / statue 85 / great_wall 79），md5+pHash 去重后无完全重复
- [x] **AI 预标注**（YOLO-World 开放词表）：269/392 张已生成初版框（sydney/oriental 100%，tiananmen 仅 18%、eiffel 39%），已自动清理嵌套框/噪点框，抽查确认多数框质量良好
- [x] **管线验证**：split(313/39/40) → check → train 1 轮试跑在真实数据+中文路径下端到端跑通
- [ ] **人工复核（下一步，需要 hanghang）**：
  1. 下载 X-AnyLabeling win-cpu 版（解压到纯英文路径），打开 `data/labeled/images/`（导入 YOLO 格式 + `data/classes.txt`）
  2. 复核/修正所有初版框；**删除带水印的图**（爬虫结果里混有昵享网等 Stock 水印图）
  3. **重点核对 11 张已检出的天安门图**——实测发现 YOLO-World 把框打到了"人民英雄纪念碑"上而非天安门城楼，这 11 张需重标；另给 `data/labeled/_needs_manual.txt` 里 123 张未检出的图补框（重点是 tiananmen / eiffel_tower）
  4. 顺手删掉误爬的无视图（如其他建筑、海报、示意图）
- [ ] 复核后重跑 `python src/data_prep/split_dataset.py && python src/data_prep/check_dataset.py --nc 6`，再正式训练 + 推理演示

| 实测项 | 结果 |
|--------|------|
| 训练 1 epoch（256 张, 320px, batch 16, freeze 10） | **11.1 秒**（约 1.4 it/s） |
| 单图推理 @320px | **24 ms ≈ 41 FPS**（端到端；纯模型推理 6.9 ms） |
| 单图推理 @640px | 39 ms ≈ 26 FPS |
| 中文路径图片预测 | 正常 |

结论：

- 训练主线：`yolo11n + imgsz=320 + batch=16 + freeze=10 + 100 epochs`
- 数据集 300~450 张时，正式训练约 **23~34 分钟**（外推公式见方案文档 6.3 节）
- 推理（320px）：PyTorch 端到端约 41 FPS，视频演示绰绰有余；导出 OpenVINO 后还能再快约 2 倍
