from ultralytics import YOLO
import os
import time
import wandb  

wandb.init(
    project="YOLO_Training",
    name="first_training",
    config={
        "epochs": 100,
        "batch_size": 32,
        "imgsz": 640,
        "optimizer": "Adam",
        "lr0": 0.003,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "device": "cuda"
    },
    reinit=True  # 允许多次运行
)

def train_yolo():
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 训练模型
    model.train(
        data="Dataset/data.yaml",  
        epochs=100,
        batch=32,
        imgsz=640,
        optimizer="Adam",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        device="cuda",  
        project="runs/WB_experiment",  
        name="first_training"
    )

    return model

if __name__ == "__main__":
    start_time = time.time()

    # 训练 YOLO
    model = train_yolo()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spent on Training: {elapsed_time:.2f} seconds")

    wandb.log({"Training Time": elapsed_time})

    metrics = model.val(data="Dataset/data.yaml", split="val")

    precision = metrics.results_dict.get("metrics/precision(B)", 0.0)
    recall = metrics.results_dict.get("metrics/recall(B)", 0.0)
    mAP50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
    mAP50_95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)
    wandb.log({
        "precision": precision,
        "recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95
    })

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")

    wandb.finish()
