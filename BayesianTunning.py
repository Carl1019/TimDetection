import time
import wandb
from ultralytics import YOLO
import optuna
import optuna.visualization as vis

# 初始化 wandb
wandb.init(project="yolov8_hpo", name="Optuna_Search", config={})

# 记录所有实验结果
results_list = []

# 设定搜索空间
search_space = {
    "lr0": (1e-4, 0.05),  # 对数空间
    "batch": [32, 64,],
    "optimizer": ["SGD", "Adam", "RMSprop"],
}

def objective(trial):
    # 采样超参数
    params = {
        "lr0": trial.suggest_float("lr0", *search_space["lr0"], log=True),
        "batch": trial.suggest_categorical("batch", search_space["batch"]),
        "optimizer": trial.suggest_categorical("optimizer", search_space["optimizer"]),
    }

    print(f"\n------------------------- Trial {trial.number} Start -------------------------")
    print(f"lr={params['lr0']:.4f}, batch={params['batch']}, optimizer={params['optimizer']}")

    # 训练模型
    model = YOLO("yolov8s.pt")
    model.train(
        data="Dataset/data.yaml",
        epochs=50,
        batch=params["batch"],
        lr0=params["lr0"],
        optimizer=params["optimizer"],
        device="cuda",
        project="runs/bayesian_search",
        name=f"trial_{trial.number}",
        verbose=False
    )

    # 评估模型
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

    # 记录到 wandb
    wandb.log({
        "trial_id": trial.number,
        "lr0": params["lr0"],
        "batch_size": params["batch"],
        "optimizer": params["optimizer"],
        "precision": trial_result["precision"],
        "recall": trial_result["recall"],
        "mAP50": trial_result["mAP50"],
        "mAP50-95": trial_result["mAP50-95"]
    })

    return trial_result["mAP50-95"]  # 目标函数优化 mAP50-95

if __name__ == "__main__":
    start_time = time.time()

    # 使用贝叶斯优化 + 提前终止无效试验
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    # 运行 50 轮优化
    study.optimize(objective, n_trials=50)

    end_time = time.time()
    print(f"\nTotal tuning time: {end_time - start_time:.2f} seconds")

    # 输出最佳结果
    best_trial = study.best_trial
    print("\n** Best Trial **")
    print(f"Trial ID: {best_trial.number}")
    print(f"Params: {best_trial.params}")
    print(f"mAP50-95: {best_trial.value:.4f}")

    # 记录实验数据
    study.trials_dataframe().to_csv("optuna_results.csv", index=False)

    # 可视化超参数优化过程
    fig_param_importance = vis.plot_param_importances(study)
    fig_optimization_history = vis.plot_optimization_history(study)

    # 上传到 wandb
    wandb.log({"param_importance": wandb.Image(fig_param_importance)})
    wandb.log({"optimization_history": wandb.Image(fig_optimization_history)})

    # 结束 wandb 运行
    wandb.finish()
