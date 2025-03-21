from ultralytics import YOLO

def test_yolo():
    # 加载训练完成的模型
    model = YOLO(r"D:\SYDE770 Project\TimDetection\runs\detect\First_success_tranining\weights\best.pt")
    # 在测试集上评估模型
    metrics = model.val(data="D:/SYDE770 Project/TimDetection/Dataset/data.yaml", split="test")
    print(metrics) 

if __name__ == "__main__":
    test_yolo()
