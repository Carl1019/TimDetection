# TimDetection
Process Description:
1. Upload the dataset
    1. Problem:some file has long path which can not be resolved by windows, so I just delete it(two images and two labels).
    2. Delete roboflow related stuff in the data.yaml
2. Create traning codes
    1. forbidden mandb
    2. set epoches to 50
        1. After 40 rounds, there is a command: Closing dataloader mosaic, ignore it, let training keep going
    3. Todo tunning!
3. TODO...

Configuration:
1. pip YOLO v8: 
    1. pip install ultralytics wandb opencv-python matplotlib
2. pip CUDA GPU
    1. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
3. if accidently interrupt:
    1. python train.py --resume

Problems:
1. mandb virtualize tool use or not? prob use for virtualiza, and shows in report.
2. watch out the file path in data.yaml(better use absolute path for all path!)
3. When running the script, I found that there is a multi-process error, which be solved by use(if __name__ == "__main__":)
4. Does 50 epoches too many?


Plans:
1. use part of the original dataset try to train our model before we are going to use better GPUs(no need, don't want the hassle of changing files, and I tried to use the whole dataset and the training speed is not that slow)
2. use a new file to detect the metrics of the model(mAP、percision、recall)
3. use a new file to test the testset
4. try to use mandb to virtualize the traning.