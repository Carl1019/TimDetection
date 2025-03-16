import random
from ultralytics import YOLO
import time

def tunning():
    # 加载训练完成的模型
    model = YOLO("yolov8s.pt")

    # 设定搜索范围
    search_space = {
        "lr0": [0.01, 0.005, 0.001, 0.0005],
        "batch": [8, 16, 32],
        "optimizer": ["SGD", "Adam", "RMSprop"]
    }

    # 随机选择 N 组参数
    i = 1
    for _ in range(15):  # 运行 15 组实验
        params = {key: random.choice(values) for key, values in search_space.items()}

        print(f"------------------------------------------------------------------------------------")
        print(f"第{i}th 训练: lr={params['lr0']}, batch={params['batch']}, optimizer={params['optimizer']}")
        print(f"------------------------------------------------------------------------------------")

        model.train(
            data=r"TimDetection/Dataset/data.yaml",
            epochs=20,
            batch=params["batch"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            device="cuda",
            project="runs/random_search",
            name=f"lr{params['lr0']}_batch{params['batch']}_{params['optimizer']}",
        )


if __name__ == "__main__":
    start_time = time.time()
    tunning()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spend on Tunning: {elapsed_time:.2f} second")