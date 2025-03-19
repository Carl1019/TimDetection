import time
from ultralytics import YOLO
import optuna

# 记录所有实验结果
results_list = []

# 设定搜索范围（不再需要预定义全量参数）
search_space = {
    "lr0": (1e-4, 0.1),          # 对数空间采样
    "batch": [8, 16, 32, 64, 128],     # 分类参数
    "optimizer": ["SGD", "Adam", "RMSprop"],
    "weight_decay": (1e-4, 1e-3)  # 对数空间采样
}

def objective(trial):
    # 自动建议超参数
    params = {
        "lr0": trial.suggest_float("lr0", *search_space["lr0"], log=True),
        "batch": trial.suggest_categorical("batch", search_space["batch"]),
        "optimizer": trial.suggest_categorical("optimizer", search_space["optimizer"]),
        "weight_decay": trial.suggest_float("weight_decay", *search_space["weight_decay"], log=True),
    }

    print(f"\n-----------------------------------------------------------------------------------------------------------------")
    print(f" Trial {trial.number} start: lr={params['lr0']:.4f}, batch={params['batch']}, optimizer={params['optimizer']}, weight_decay={params['weight_decay']:.4f}")
    
    # 模型训练
    model = YOLO("yolov8s.pt")
    model.train(
        data="Dataset/data.yaml",
        epochs=100,
        batch=params["batch"],
        lr0=params["lr0"],
        optimizer=params["optimizer"],
        weight_decay=params["weight_decay"],
        device="cuda",
        project="runs/bayesian_search",
        name=f"trial_{trial.number}",
        verbose=True  # 减少输出冗余
    )

    # 模型验证
    metrics = model.val(data="Dataset/data.yaml", split="val")
    
    # 记录结果
    trial_result = {
        "trial_id": trial.number,
        "params": params,
        "precision": metrics.results_dict["metrics/precision(B)"],
        "recall": metrics.results_dict["metrics/recall(B)"],
        "mAP50": metrics.results_dict["metrics/mAP50(B)"],
        "mAP50-95": metrics.results_dict["metrics/mAP50-95(B)"]
    }
    results_list.append(trial_result)
    
    return trial_result["mAP50-95"]  # 优化目标设为mAP50-95

if __name__ == "__main__":
    start_time = time.time()
    
    # 创建Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42))  # 使用TPE算法
    
    # 运行优化
    study.optimize(objective, n_trials=15)
    #study.optimize(objective, n_trials=30)
    #study.optimize(objective, n_trials=50)

    end_time = time.time()
    print(f"\nTotal tuning time: {end_time - start_time:.2f} seconds")

    # 输出最佳结果
    best_trial = study.best_trial
    print("\n** Best trial **")
    print(f"Trial ID: {best_trial.number}")
    print(f"Params: {best_trial.params}")
    print(f"mAP50-95: {best_trial.value:.4f}")

    # 各指标最优解
    print("\n** All metrics best **")
    print(f"Best precision: {max(r['precision'] for r in results_list):.4f}")
    print(f"Best recall: {max(r['recall'] for r in results_list):.4f}")
    print(f"Best mAP50: {max(r['mAP50'] for r in results_list):.4f}")
    print(f"Best mAP50-95: {max(r['mAP50-95'] for r in results_list):.4f}")