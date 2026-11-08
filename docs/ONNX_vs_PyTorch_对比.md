# YOLO11 `.pt` 导出 ONNX：准确度与优缺点实测对比

> 实测日期：2026-11-07 ｜ 环境：`D:\Anaconda\envs\yolo_env`（Python 3.9.25 / torch 2.8.0+cpu / ultralytics 8.4.21 / onnx 1.17.0 / onnxruntime 1.19.2）
> 硬件：AMD Ryzen 7 7840H（纯 CPU，8 线程）｜ 模型：yolo11n @320，5 类地标建筑

---

## 一、结论速览

**导出 ONNX 没有任何精度损失。** 在同一套 34 张 test 集、同一 `imgsz=320`、同一输入管线（`rect=False`）下：

| 指标 | 原 `.pt` | FP32 重存 `.pt` | ONNX | 差异 |
|---|---|---|---|---|
| Precision | 0.7607 | 0.7607 | 0.7607 | 0 |
| Recall | 0.9087 | 0.9087 | 0.9087 | 0 |
| **mAP@0.50** | **0.8684** | **0.8684** | **0.8684** | **0** |
| mAP@0.50:0.95 | 0.5388 | 0.5388 | 0.5388 | 0 |

- 逐类 AP@50 五个类别**全部相同**（0.9950 / 0.7816 / 0.9950 / 0.8283 / 0.7423）。
- 同一输入张量下网络原始输出 `(1, 9, 2100)`：最大绝对误差 `9.46e-4`、平均 `1.55e-5`（算子融合/浮点累加顺序造成的舍入级误差）。
- 最终检测框：38/38 **100% 匹配**，置信度平均差 `0.000000`，坐标平均差 `0.0001 px`。
- CPU 端到端单图延迟：`41.7 ms → 31.4 ms`，**1.33×**（24.0 → 31.9 FPS）。
- 体积：`5.21 MB → 10.00 MB`（约 2 倍，原因见第四节）。

一句话：**换 ONNX 是"免费的精度 + 小幅提速 + 部署自由度"，代价是文件变大、不能训练、尺寸写死。**

---

## 二、导出操作与产物

导出脚本：`src/inference/export_onnx.py`（与训练参数 `args.yaml` 一致，`imgsz=320`）。

```bash
python src/inference/export_onnx.py
# 可选：--imgsz 640 / --dynamic / --nms / --opset 17
```

产物：`runs/detect/landmark_v5_1025/weights/best.onnx`

| 项目 | 值 |
|---|---|
| 输入 | `images: [1, 3, 320, 320]` float32，静态 batch |
| 输出 | `output0: [1, 9, 2100]`（4 个框坐标 + 5 个类别分数，**未含 NMS**） |
| opset / IR | 17 / 8 |
| 节点数 | 318（onnxslim 0.1.87 化简后） |
| 元数据 | 类别名、stride=32、imgsz=[320,320]、task=detect、batch=1 均写入图 |
| 校验 | `onnx.checker.check_model` 通过 |

已在现有推理脚本里加好 `--onnx` 开关（自动把 `.pt` 路径换成同目录 `.onnx`）：

```bash
python src/inference/predict_image.py --onnx
# 实测输出：eiffel-tower-test.jpg: eiffel_tower (conf=0.92) box=[258.1, 408.5, 590.0, 1138.9]
```

也可以直接 `YOLO(".../best.onnx").predict(...)`，Ultralytics 会自动切到 ONNX Runtime 后端。视频脚本同样支持 `--onnx`。

### 纯 ONNX Runtime 部署（不装 PyTorch / Ultralytics）

`src/inference/predict_onnx_native.py` 只依赖 `numpy + opencv-python + onnxruntime`，
自己实现 letterbox（含 BGR→RGB）、xywh→xyxy 解码、按类别 NMS、坐标还原，
类别名与输入尺寸直接读 ONNX 的 metadata_props，不另带配置文件：

```bash
python src/inference/predict_onnx_native.py --save
python src/inference/predict_onnx_native.py --source data/dataset/images/test --save
# 结果图 -> runs/detect/onnx_native/
```

在 34 张 test 上与 `YOLO(best.onnx)` 逐图对比（`compare/verify_native_onnx.py` 可复现）：
**35/35 框 100% 匹配、置信度差 0、坐标平均差 0.54 px**（letterbox 取整差异），单图纯推理约 20 ms。
实测测试图：`eiffel_tower conf=0.917 box=[256.1, 408.5, 588.0, 1138.9]`。

> 写这类脚本最容易漏的是 **BGR→RGB 翻转**（Ultralytics 的 `preprocess` 会做 `im[..., ::-1]`）：
> 漏掉会得到 conf 0.785、框明显内缩的错误结果；补上后与官方管线完全对齐。

> **注意**：`.onnx` 输出是原始 head 结果，若脱离 Ultralytics 用纯 `onnxruntime` 部署，需要自己实现 letterbox 预处理、DFL/框解码和 NMS。想让 NMS 进图，可重新导出时加 `--nms`（输出变为 `[1, 300, 6]`）。

---

## 三、⚠️ 一个会骗到人的坑：`rect` / letterbox

**最初直接对比时得到 `PT 0.8221` vs `ONNX 0.8684`，看起来"ONNX 更准 0.046"，这是假象。**

原因（已用 `compare/probe_pipeline.py` 抓到真实输入张量坐实）：

| | 输入张量 | 输出 anchor 数 | mAP@50 |
|---|---|---|---|
| `.pt` 默认（`rect=True`） | `(1,3,320,256)` 矩形 letterbox | 1680 | 0.8221 |
| ONNX（静态图，被强制 `rect=False`） | `(1,3,320,320)` 正方形 letterbox | 2100 | 0.8684 |
| `.pt` 显式 `rect=False` | `(1,3,320,320)` 正方形 | 2100 | **0.8684** |

Ultralytics 对 PyTorch 模型默认启用矩形 letterbox（按原图长宽比少填充，省算力），但静态尺寸的 ONNX 图只接受固定的 320×320，`validator.py` 会强制 `rect=False`。两者喂进网络的像素分布不同，框坐标和置信度自然有偏移；在只有 34 张图 / 36 个目标的小测试集上，这种偏移被放大成 0.046 的 mAP 差。

**对照实验**：给 `.pt` 显式传 `rect=False` 后，P / R / mAP 与 ONNX 逐位相同（0.7607 / 0.9087 / 0.8684）。所以：

> 评估不同格式模型时，必须先把输入预处理对齐，否则测到的是管线差异，不是模型/框架差异。
> 本项目所有对比脚本已强制 `rect=False`；若要用动态输入，可 `--dynamic` 重新导出 ONNX。

---

## 四、为什么 ONNX 文件大一倍

`best.pt` 里的权重是 **FP16 序列化**（418 个 float16 张量），Ultralytics 加载时自动 `.float()` 转成 FP32 再推理；ONNX 导出按 FP32 落盘（181 个 float32 initializer）。所以：

| 文件 | 大小 | 说明 |
|---|---|---|
| `best.pt` | 5.21 MB | FP16 存储 + pickle 元数据；加载后是 FP32 计算 |
| `best_fp32.pt`（对照用） | 10.20 MB | 同权重重存为 FP32，实测输出与原 `.pt` **完全一致**（误差 0） |
| `best.onnx` | 10.00 MB | FP32 静态图，含算子/结构信息 |

想要小体积可导出 `half=True` 的 FP16 ONNX：**本机实测体积减半到 5.05 MB、mAP 仅差 0.0002，但 CPU 延迟从 27.97 ms 涨到 50.45 ms**，纯 CPU 不划算，GPU / TensorRT / 分发受限场景才值得。四种导出变体的完整实测见第八节。

---

## 五、优缺点对照

| 维度 | PyTorch `.pt` | ONNX Runtime `.onnx` |
|---|---|---|
| **准确度** | mAP@50 **0.8684** | **0.8684**（零损失，框 100% 一致） |
| **训练 / 微调 / 再导出** | ✅ 原生支持，可继续 train、改类别、量化 | ❌ 只读推理图，不能反向传播 |
| **CPU 推理速度** | 41.7 ms / 24.0 FPS | **31.4 ms / 31.9 FPS（1.33×）** |
| **文件体积** | **5.21 MB** | 10.00 MB |
| **运行时依赖** | PyTorch + ultralytics（数百 MB～GB） | onnxruntime（几十 MB）即可 |
| **部署形态** | 基本限 Python | C++/C#/Java/JS、移动端、Web、TensorRT/OpenVINO 二次转换 |
| **输入灵活性** | 任意尺寸、矩形 letterbox、动态 batch | 本次为静态 `[1,3,320,320]`，换分辨率需重导或 `--dynamic` |
| **工具链** | `val` / `train` / `benchmark` / 混淆矩阵全套 | 加载进 Ultralytics 仍可用 `val`；纯 ONNX 需自写前后处理 |
| **生态风险** | pickle 反序列化安全、torch 版本兼容 | opset / 算子兼容、类别名与 NMS 需自行维护 |
| **加速上限** | 一般 | 可继续转 OpenVINO / TensorRT / INT8 量化 |

### `.pt` 的优点
1. 训练链路完整，可继续微调、改数据后重训。
2. 体积小（FP16 序列化），原生支持动态尺寸与 `rect` 矩形推理。
3. Ultralytics 官方工具链开箱即用，`val`/`plots`/`benchmark` 全都有。
4. 想换导出格式（ONNX / OpenVINO / TensorRT）都得从它出发，**必须保留**。

### `.pt` 的缺点
1. 部署必须带 PyTorch，环境重、启动慢、跨语言几乎不可行。
2. CPU 端到端比 ONNX 慢约 33%。
3. pickle 权重存在反序列化安全风险，torch/ultralytics 版本变动可能导致加载失败。

### ONNX 的优点
1. **精度零损失**：本任务 mAP、逐类 AP、检测框、置信度全部对齐。
2. CPU 提速 1.33×，且无需 PyTorch；静态图便于图优化、算子融合。
3. 框架中立，一次导出可投到多种运行时，还能再转 OpenVINO / TensorRT / INT8。
4. 类别名、stride、输入尺寸等元数据已写进图，便于独立交付。

### ONNX 的缺点
1. 不能训练，永远只是"派生产物"。
2. 文件约 2 倍大（FP32 落盘）。
3. 静态尺寸 + batch=1，`rect` 矩形 letterbox 用不了；换分辨率要重导或 `--dynamic`。
4. 输出是 NMS 前的原始张量，脱离 Ultralytics 要自己写预处理 + 解码 + NMS。本项目已提供现成实现 `predict_onnx_native.py`（纯 numpy/cv2/onnxruntime，见第二节），但仍需随模型结构变化自行维护。
5. opset / 算子兼容性需要在目标设备上再验一次；图化简（onnxslim）会带来 `~1e-3` 级数值抖动，虽不影响结果，但做逐比特校验时要注意。

---

## 六、选型建议

| 场景 | 建议 |
|---|---|
| 继续训练、调参、换数据集 | `.pt` |
| 本项目 Python 演示 / 写报告复现 | 两者都行；`.pt` 工具链更顺手 |
| 纯 CPU 服务器批量推理、要更快 | **ONNX**（1.33×） |
| 交付给没有 PyTorch 的环境 / C++ / 移动端 / Web | **ONNX** |
| 要上 Intel CPU 极限加速 / GPU TensorRT | 以 ONNX 为中间格式再转 OpenVINO / TensorRT |
| 输入分辨率经常变 | `.pt`，或重新导出 `--dynamic` 的 ONNX |

**推荐做法：两个都留。** `.pt` 作为可再训练的主权重，`best.onnx` 作为部署产物；每次重训后重新跑一遍导出 + 对比脚本，确认精度没有回退。导出参数（`--dynamic` / `--nms` / `--half`）怎么选，见第八节的四变体实测。

---

## 七、复现命令与文件清单

```bash
# 1) 导出（FP32, 静态 320, opset 17, onnxslim 化简）
python src/inference/export_onnx.py

# 2) 三方对比：准确度 / 数值误差 / 检测框一致性 / 体积 / CPU 延迟
python src/train_eval/compare_pt_onnx.py
#    结果 -> runs/detect/landmark_v5_1025/compare/results.json

# 3) 用 ONNX 推理
python src/inference/predict_image.py --onnx
python src/inference/predict_video.py --source data/demo.mp4 --onnx

# 4) 纯 ONNX Runtime 部署（不依赖 PyTorch / Ultralytics）
python src/inference/predict_onnx_native.py --save
python runs/detect/landmark_v5_1025/compare/verify_native_onnx.py   # 与官方管线逐框校验

# 5) 复现 rect 陷阱（可选）
python runs/detect/landmark_v5_1025/compare/probe_pipeline.py

# 6) 导出变体实测（dynamic / nms / half）
python src/train_eval/benchmark_onnx_variants.py

# 7) NMS 入图版的纯部署与校验
python src/inference/predict_onnx_nms_native.py --save
python runs/detect/landmark_v5_1025/compare/variants/verify_nms_native.py
```

| 文件 | 作用 |
|---|---|
| `runs/detect/landmark_v5_1025/weights/best.onnx` | **导出的 ONNX 模型（最终产物）** |
| `src/inference/export_onnx.py` | 导出 + `onnx.checker` 校验脚本 |
| `src/train_eval/compare_pt_onnx.py` | 三方对比评测脚本 |
| `src/train_eval/benchmark_onnx_variants.py` | 四种导出变体的体积 / mAP / 延迟实测 |
| `runs/detect/landmark_v5_1025/compare/variants/results.json` | 变体实测数据 |
| `src/inference/predict_onnx_native.py` | **纯 ONNX Runtime 端到端推理（无 torch/ultralytics）** |
| `src/inference/predict_onnx_nms_native.py` | **NMS 入图版的纯部署脚本（无需自写 NMS）** |
| `runs/detect/landmark_v5_1025/compare/variants/verify_nms_native.py` | NMS 版与官方管线逐框校验 |
| `runs/detect/landmark_v5_1025/compare/verify_native_onnx.py` | 无 NMS 版与官方管线逐框校验 |
| `runs/detect/onnx_native/eiffel-tower-test.jpg` | 纯 ONNX 推理画框结果图 |
| `runs/detect/onnx_nms_native/eiffel-tower-test.jpg` | NMS 入图版画框结果图 |
| `src/inference/predict_image.py` / `predict_video.py` | 新增 `--onnx` 开关 |
| `runs/detect/landmark_v5_1025/compare/results.json` | 全部实测数据 |
| `runs/detect/landmark_v5_1025/compare/data_local.yaml` | 对比脚本自动生成的数据集配置副本 |
| `runs/detect/landmark_v5_1025/compare/best_fp32.pt` | 变量控制用的 FP32 对照权重 |
| `runs/detect/landmark_v5_1025/compare/probe_pipeline.py` | 定位 `rect` 差异的诊断脚本 |

> **已修复（2026-11-08）**：项目根的 `data.yaml` 里 `path` 原是迁移前的
> `D:/桌面/YOLO训练识别/data/dataset`，重训练 / 直接 `yolo val` 会踩坑。
> 现给 `src/data_prep/split_dataset.py` 加了 `--yaml-only` 参数（只改 path，不重新划分数据集），
> 已重新生成 `data.yaml` 为 `D:/YOLO训练识别/data/dataset`，类别定义与划分保持不变；
> 用修复后的配置直接 `val(best.onnx)` 复验 **mAP@50 = 0.8684**。

---

## 八、导出变体实测（2026-11-08 补测）

同一份 `best.pt` 换不同导出参数各生成一份 ONNX，全部落在 `compare/variants/`，**不覆盖主产物**。
脚本：`src/train_eval/benchmark_onnx_variants.py`。
条件：test 34 张 / `imgsz=320` / `rect=False` / CPU；延迟取 20 张图 × 2 轮的中位数。

| 变体 | 体积 | 输入 → 输出 | 权重精度 | mAP@50 | 中位延迟 | 结论 |
|---|---|---|---|---|---|---|
| FP32 静态（**主产物**） | 10.00 MB | `[1,3,320,320]` → `[1,9,2100]` | float32 | 0.8684 | 27.97 ms | 基准 |
| `--dynamic` | 10.34 MB | `['batch',3,'height','width']` → `['batch',9,'anchors']` | float32 | 0.8684 | 47.28 ms | 精度不变，CPU 慢 1.69× |
| `--nms` | 10.03 MB | `[1,3,320,320]` → `[1,300,6]` | float32 | 0.8793 * | **23.65 ms** | 最快，NMS 已在图内 |
| `--half` | **5.05 MB** | `[1,3,320,320]` → `[1,9,2100]` | float16 | 0.8682 | 50.45 ms | 体积减半、精度几乎无损，CPU 慢 1.80× |

> 延迟为本批次内部横向比较（20 张子集），与前面 41.7 / 31.4 ms 那批（34 张）绝对值不可直接横比。
>
> \* `0.8793` 是**后处理口径差异，不是模型变准**：图内 NMS 在导出时按默认 conf=0.25 过滤，
> 而 `val` 的外置 NMS 用 conf=0.001；test 仅 34 张，0.011 也在小样本波动范围。
> 严格比精度必须统一 conf / iou 口径。

### 逐项解读

- **FP16（`--half`）**：之前"CPU 上 FP16 支持差"的推测不成立——本机 onnxruntime 1.19.2 能正常加载、
  跑通 val 与 predict，mAP 只差 0.0002（舍入级），体积 10.00 → 5.05 MB。
  但 CPU 需要 FP16↔FP32 转换，延迟 27.97 → 50.45 ms，**纯 CPU 部署不划算**；
  适合分发/存储受限，或 GPU / TensorRT 场景。
- **动态尺寸（`--dynamic`）**：输入变成符号维度，节点数 318 → 417（多出 shape 推导常量）。
  mAP 完全相同（本任务强制 `rect=False`，没用上动态能力），CPU 却慢 1.69×——动态 shape 吃不到静态图优化。
  只有"同一模型要跑多种分辨率 / batch"时才值得，且更建议在 GPU 上用。
- **NMS 入图（`--nms`）**：输出变成 `[1,300,6]`（x1,y1,x2,y2,conf,cls，坐标仍是 320 尺度，需自行还原），
  省掉 Python 侧 NMS，端到端 **23.65 ms**，比静态基线快 1.18×、比 PyTorch 41.7 ms 快约 1.76×。
  纯 onnxruntime 直跑实测首框 `[117.5, 102.1, 200.5, 284.7, 0.92, 0]`（eiffel_tower）。
  **最推荐给"脱离 Ultralytics 的纯部署"**：不用自己写 NMS，只需 letterbox + 坐标还原。
  已提供配套脚本 `src/inference/predict_onnx_nms_native.py`（复用纯 numpy 预处理函数），
  34 张 test 与 `YOLO(nms_onnx)` 逐框校验 **38/38 匹配、conf 差 0、坐标差 0.52 px**
  （`compare/variants/verify_nms_native.py`）；测试图输出
  `eiffel_tower conf=0.917 box=[256.1, 408.5, 588.0, 1138.9]`，与无 NMS 版结果一致。
  注意 `predict_onnx_native.py` 是针对无 NMS 版本写的，两个脚本不能混用模型。

### 选型补充

| 场景 | 推荐变体 |
|---|---|
| 现有项目、全功能验证 / 复现报告 | 主产物 FP32 静态 |
| 纯部署、要最快、想省掉自写 NMS | `--nms` |
| 分发体积敏感、目标设备是 GPU / 移动端 | `--half`（5.05 MB） |
| 需要多分辨率 / 动态 batch | `--dynamic`（注意 CPU 会变慢） |
| 纯 CPU 生产 | 别用 `--half` 和 `--dynamic` |

> 复现：`python src/train_eval/benchmark_onnx_variants.py`；
> 结果 `compare/variants/results.json`，三份变体模型在 `compare/variants/*/model.onnx`。
