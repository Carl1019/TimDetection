from ultralytics import YOLO
import os
import time

# 暂时禁用 wandb
os.environ["WANDB_DISABLED"] = "true"

def train_yolo():
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 训练参数
    model.train(
        data="Dataset/data.yaml",  
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.001,
        weight_decay=0.0005,
        dropout=0.1,
        device="cuda"  # 使用我自己的GPU
    )
if __name__ == "__main__":
    start_time = time.time()
    train_yolo()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spend on Training: {elapsed_time:.2f} second")

