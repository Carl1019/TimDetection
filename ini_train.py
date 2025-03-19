from ultralytics import YOLO
import time
import wandb  

# 初始化 W&B 项目
wandb.init(
    project="YOLOv8_Training_Nebula",  # 你的 W&B 项目名称
    name="Nebula11_first_training",  # 本次实验的名称
    config={  # 记录超参数
        "epochs": 10,
        "batch_size": 32,
        "image_size": 640,
        "optimizer": "SGD",
        "learning_rate": 0.003,
        "lr_final": 0.01,
    }
)

def train_yolo():
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 启用 W&B 监控
    wandb_callback = wandb.run.dir  # W&B 会自动监听 ultralytics 训练数据

    # 训练参数
    model.train(
        data="Dataset/data.yaml",
        epochs=wandb.config.epochs,
        batch=wandb.config.batch_size,
        imgsz=wandb.config.image_size,
        optimizer=wandb.config.optimizer,
        lr0=wandb.config.learning_rate,
        lrf=wandb.config.lr_final,
        device="cuda",  # 使用 GPU
        project="runs/Nebula11_experiment1",
        name="first_training_with_Nebula-11_metrics",
        callbacks=[wandb_callback]  # **启用 W&B 监控**
    )

if __name__ == "__main__":
    start_time = time.time()
    train_yolo()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spent on Training: {elapsed_time:.2f} seconds")

    # 记录训练时间到 W&B
    wandb.log({"Training Time (s)": elapsed_time})

    # 结束 W&B 运行
    wandb.finish()
