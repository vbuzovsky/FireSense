# My Fork of YOLOv7 (WongKinYiu/yolov7) for my bachelor thesis

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
Original repository: [YOLOv7 ](https://github.com/WongKinYiu/yolov7)


## Installation

To be able to run inference with fire & smoke detection, you need to download the weights from [Semestral's output](https://drive.google.com/drive/u/0/folders/10fE3ess1fwAso3T3bpbyl2NOF1YlS3gR)
   
Install the requirements:
```bash
   $ pip install -r requirements.txt
```


## Added main features:
 - [x] Calculating average bbox for each class
 - [x] Cutting out the bboxes from the original frame and saving them as images
 - [x] Calculating optical flow of whole IMG_BUFFER (detect.py)
 - [x] Cutting part of full optical flow to match average bbox