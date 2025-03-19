import wandb
wandb.init(project="YOLO_Tim", name="test_run",entity="993657526-university-of-waterloo")
wandb.log({"test": 1})
wandb.finish()