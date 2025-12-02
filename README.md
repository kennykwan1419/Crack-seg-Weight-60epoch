博客LINK:https://blog.csdn.net/qq_42963855/article/details/155494360?sharetype=blogdetail&sharerId=155494360&sharerefer=PC&sharesource=qq_42963855&spm=1011.2480.3001.8118


# 📌 Crack-Seg YOLOv11-Seg 第一轮训练权重（Stage-1 Pretrain）

本仓库公开我在 **CRACK-SEG** 第一阶段预训练得到的模型权重 `best.pt`。
该权重旨在提供 **裂缝分割（crack segmentation）** 的基础能力，自定义数据微调的起点。

> 📌 本仓库仅包含第一轮预训练结果。
---

# 📦 1. 目录结构

```
CrackSeg-Stage1/
│
├── best.pt                     # 第一轮预训练模型权重
├── last.pt                     # 最后一轮预训练模型权重
├── train_yolo_stage1_crack.yaml# 训练配置文件（YOLO）
├── sample_infer/               # 推理示例
└── README.md                   # 当前文件
```

🔸 单张图片推理
```
  python infer_crack.py --model best.pt --source test.jpg
```

---

# ⚙️ 2. 环境依赖（Dependencies）

本模型基于 **Ultralytics YOLO（YOLOv8/YOLOv11 Segmentation）**。

建议使用：

| 依赖          | 版本                   |
| ----------- | -------------------- |
| Python      | 3.9–3.11             |
| PyTorch     | ≥ 2.1（with CUDA）     |
| Ultralytics | ≥ 8.1.0              |
| CUDA        | 11.8 / 12.x          |
| GPU         | 6GB VRAM 以上（推荐 8GB+） |

### 🔧 2.1 创建虚拟环境（可选）

```bash
python -m venv .venv
source .venv/bin/activate
```

### 🔧 2.2 安装 Ultralytics

```bash
pip install ultralytics
```

如果你需要更快推理：

```bash
pip install onnxruntime-gpu
```

如需导出 TensorRT：

```bash
pip install tensorrt
```

---

# 🧩 3. 数据集结构（如需复现训练）

第一轮使用公开数据集 **Crack-Seg**，你需要准备如下结构：

```
datasets/
  crackseg/
    images/
       train/*.jpg
       val/*.jpg
    labels/
       train/*.png    # segmentation masks
       val/*.png
```

mask 必须是 **二值分割图**（0/1 或黑白）。

---

# 🚀 4. 模型训练（Train）

使用 YOLOv11-Seg 进行第一轮预训练：

```bash
yolo segment train \
  model=yolo11s-seg.pt \
  data=train_yolo_stage1_crack.yaml \
  epochs=100 \
  imgsz=416 \
  batch=16 \
  device=0 \
  amp=True \
  cache=True
```

### 说明：

* `yolo11s-seg.pt` 为官方预训练 segmentation 模型
* `train_yolo_stage1_crack.yaml` 在本仓库内
* 推荐使用 **1024×1024** 输入尺寸，稳定且泛化较好
* `cache=True` 能有效提升训练 I/O 性能

---

# 🔍 5. 模型推理（Inference）

你可以直接使用 `best.pt` 对任意墙体裂缝图片进行 segmentation 推理。

### 单张推理：

```bash
yolo segment predict \
  model=best.pt \
  source=your_image.jpg \
  save=True \
  imgsz=1024
```

预测结果会保存在：

```
runs/segment/predict/
```

---

# 🧪 6. Python API 推理示例（推荐）

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("best.pt")

# 推理
results = model("test.jpg")

# 遍历 mask
for r in results:
    masks = r.masks.data  # segmentation masks
    boxes = r.boxes       # bounding boxes (if enabled)
    print("Masks:", masks.shape)
```

---

# 📝 7. YAML 配置文件说明（train_yolo_stage1_crack.yaml）

```yaml
path: datasets/crackseg

train: images/train
val: images/val

names:
  0: crack

task: segment

imgsz: 416
epochs: 100
batch: 16
optimizer: SGD
lr0: 0.01

augment:
  hsv: 0.015
  flipud: 0.0
  fliplr: 0.5
  mosaic: 0.1
  blur: 0.2
```

### 参数解释：

* `imgsz: 1024`
  更适合裂缝细节，不会过度压缩

* `mosaic: 0.1`
  适合 crack 线状缺陷，不会切得太离谱

* `flip` 系列增强适合 crack pattern

---

# 🎯 8. 使用场景（Why you should use this weight）

本预训练权重适用于：

* 建筑外墙裂缝检测
* 路面裂缝检测
* 地面混凝土裂缝
* 高分辨率工业缺陷
* 后续 Fine-tune（1536×1536 / 1920×1920）
* 多机房分布式推理

在我的测试中，该模型作为第一轮预训练权重能大幅提升：

| 指标          | 提升趋势            |
| ----------- | --------------- |
| seg_loss    | ↓ 降低约 60%       |
| recall      | ↑ 提升约 20–30%    |
| mask mIoU   | ↑ 提升约 0.07–0.15 |
| 不同墙体纹理的泛化能力 | ↑ 明显提升          |

---

# 📤 9. 模型导出（可转 ONNX / TensorRT）

### ONNX 导出

```bash
yolo export model=best.pt format=onnx opset=12
```

### TensorRT 导出

```bash
yolo export model=best.pt format=engine half=True
```

可直接用于：

* C++ 推理
* Java pipeline
* NVIDIA Jetson
* Triton server


# 📫 联系

如果你在使用过程中遇到其它技术问题欢迎通过CSDN私信联络我。


