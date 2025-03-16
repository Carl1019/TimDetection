import wandb

wandb.init(project="wandb_test_project", name="test_run")

wandb.log({"accuracy": 0.95, "loss": 0.05})

wandb.finish()

