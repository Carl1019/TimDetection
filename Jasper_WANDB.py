from ultralytics import YOLO
import os
import time
import wandb 

Project_Name = "YOLO_Tim"
Run_Name = "training test 11"


def on_train_epoch_end(trainer):
    run.log({"train": trainer.metrics})


def train_yolo():
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 训练参数
    # model.train(
    #     data="Dataset/data.yaml",  
    #     epochs=5, #测试用的话把这个调小
    #     batch=16,
    #     imgsz=640,
    #     optimizer="SGD",
    #     lr0=0.01,
    #     lrf=0.001,
    #     weight_decay=0.0005,
    #     dropout=0.1,
    #     # device="cuda",  # 使用 GPU
    #     project="runs/WB_experiment",  
    #     name="first training",
    # )

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.train(
        data="Dataset/data.yaml",
        epochs=5,  # 只测试一个 epoch
        batch=1,  # 使用较小的 batch
        imgsz=64,  # 缩小图片尺寸
        optimizer="Adam",  # 使用更快的 Adam 优化器
        lr0=0.001,  # 调整学习率
        weight_decay=0.0005,
        dropout=0.0,  # 关闭 dropout
        device="cpu",  # 使用 CPU（如果没有 GPU）
        project="runs/WB_experiment",
        name="training test",
    )
if __name__ == "__main__":
    # 初始化 wandb
    run = wandb.init(project=Project_Name, name=Run_Name)

    start_time = time.time()
    train_yolo()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spend on Training: {elapsed_time:.2f} second")

    # 结束 W&B 运行
    run.finish()