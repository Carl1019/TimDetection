from ultralytics import YOLO
import os
import time
import wandb 

# 初始化 wandb
#wandb.init(mode="disabled")

def train_yolo():
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 训练参数
    model.train(
        data="Dataset/data.yaml",  
        epochs=100,
        batch=32,
        imgsz=640,
        optimizer="Adam",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        device="cuda",  # 使用 GPU
        project="runs/WB_experiment",  
        name="first training",
    )

if __name__ == "__main__":
    start_time = time.time()
    train_yolo()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spend on Training: {elapsed_time:.2f} second")

    # 记录训练时间到 W&B
    #wandb.log({"Training Time": elapsed_time})

    # 结束 W&B 运行
    #wandb.finish()
