# My Fork of YOLOv7 (WongKinYiu/yolov7) for my thesis

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

Original repository: [YOLOv7 ](https://github.com/WongKinYiu/yolov7)


## Installation

To be able to run inference with fire & smoke detection, you need to download the weights from [Semestral's output](https://drive.google.com/drive/u/0/folders/10fE3ess1fwAso3T3bpbyl2NOF1YlS3gR)
   
Install the requirements (preferably in a virtual environment):
```bash
   $ pip install -r requirements.txt
```

Basic inference:
```bash
   $ python detect.py --source <path_to_video> --weights <path_to_weights> --conf CONFIDENCE_VALUE
```

