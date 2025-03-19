from ultralytics import YOLO
import wandb
run = wandb.init(project="YOLO_Tim_nubula", name="test_evaluation")
def test_yolo():
    # 加载训练完成的模型
    
    model = YOLO(r"/home/student08/TimDetection/runs/WB_experiment/first training5/weights/best.pt")
    # 在测试集上评估模型
    metrics = model.val(data="/home/student08/TimDetection/Dataset/data.yaml", split="test")
    
    print(metrics) 

if __name__ == "__main__":
    test_yolo()
    run.finish()
