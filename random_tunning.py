import random
import time
from ultralytics import YOLO

# 记录所有实验结果
results_list = []

# 设定搜索范围
search_space = {
    "lr0": [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    "batch": [8, 16, 32, 64],
    "optimizer": ["SGD", "Adam", "RMSprop"],
    "weight_decay": [0.0001, 0.0005, 0.001],
}

def tunning():
    global results_list
    for i in range(15): 
        params = {key: random.choice(values) for key, values in search_space.items()}
        print(f"\n-----------------------------------------------------------------------------------------------------------------")
        print(f"\n {i+1} round training start: lr={params['lr0']}, batch={params['batch']}, optimizer={params['optimizer']}, weight_decay={params['weight_decay']}")
        print(f"\n-----------------------------------------------------------------------------------------------------------------")
        model = YOLO("yolov8s.pt")

        model.train(
            data="Dataset/data.yaml",
            epochs=50,
            batch=params["batch"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            weight_decay=params["weight_decay"],
            device="cuda",
            project="runs/random_search",
            name=f"lr{params['lr0']}_batch{params['batch']}_{params['optimizer']}",
        )
        print(f"\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"\n{i+1} round training completed")
        print(f"\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # 评估模型
        metrics = model.val(data="Dataset/data.yaml", split="val")

        # 获取不同性能指标
        precision = metrics.results_dict["metrics/precision(B)"]
        recall = metrics.results_dict["metrics/recall(B)"]
        mAP50 = metrics.results_dict["metrics/mAP50(B)"]
        mAP50_95 = metrics.results_dict["metrics/mAP50-95(B)"]

        # 记录超参数和结果
        results_list.append({
            "lr0": params["lr0"],
            "batch": params["batch"],
            "optimizer": params["optimizer"],
            "weight_decay": params["weight_decay"],
            "precision": precision,
            "recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95
        })
        print(f"\n***********************************************************************************************************************")
        print(f"\n{i+1} round finished with metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
        print(f"\n***********************************************************************************************************************")

if __name__ == "__main__":
    start_time = time.time()
    tunning()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal tuning time: {elapsed_time:.2f} seconds")

    # 找出不同性能指标下的最佳超参数
    best_precision = max(results_list, key=lambda x: x["precision"])
    best_recall = max(results_list, key=lambda x: x["recall"])
    best_mAP50 = max(results_list, key=lambda x: x["mAP50"])
    best_mAP50_95 = max(results_list, key=lambda x: x["mAP50-95"])

    print("\n**List of optimal hyperparameters:**")
    print(f"\nOptimal hyperparameters for Precision: lr0={best_precision['lr0']}, batch={best_precision['batch']}, optimizer={best_precision['optimizer']}, Precision={best_precision['precision']:.4f}")
    print(f"\nOptimal hyperparameters for Recall: lr0={best_recall['lr0']}, batch={best_recall['batch']}, optimizer={best_recall['optimizer']}, Recall={best_recall['recall']:.4f}")
    print(f"\nOptimal hyperparameters for mAP50: lr0={best_mAP50['lr0']}, batch={best_mAP50['batch']}, optimizer={best_mAP50['optimizer']}, mAP50={best_mAP50['mAP50']:.4f}")
    print(f"\nOptimal hyperparameters for mAP50-95: lr0={best_mAP50_95['lr0']}, batch={best_mAP50_95['batch']}, optimizer={best_mAP50_95['optimizer']}, mAP50-95={best_mAP50_95['mAP50-95']:.4f}")
