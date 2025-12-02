#!/bin/bash
source .venv/bin/activate

python - << 'EOF'
from ultralytics import YOLO

# 1️⃣ 载入预训练的 YOLOv11-Seg
model = YOLO("yolo11s-seg.pt")

# 2️⃣ 开始训练
model.train(
    data="configs/train_yolo_stage1_crack.yaml",
    epochs=100,
    batch=16,
    imgsz=1024,
    device=0,
    workers=8,
    amp=True,
    cache=True,
)
EOF

