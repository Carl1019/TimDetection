# TimDetection
Process Description:
1. Upload the dataset
    1. Problem:some file has long path which can not be resolved by windows, so I just delete it(two images and two labels).
    2. Delete roboflow related stuff in the data.yaml
2. Create traning codes
3. TODO...

Configuration:
1. pip YOLO v8: 
    1. pip install ultralytics wandb opencv-python matplotlib
2. 

Plan:
1. use part of the original dataset try to train our model before we are going to use better GPUs