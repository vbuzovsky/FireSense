import argparse
import time
import math
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# my imports:

import numpy as np
import my_utils.bb_average
import my_utils.snapshot_clear
import tensorflow as tf
import tensorflow.keras as keras
import queue


# -- ignore stoopid tensorflow warnings on m1 chips --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)s
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Initialize image buffer
    BUFFER_SIZE = 10
    IMG_BUFFER = queue.Queue(BUFFER_SIZE)
    OPT_FLOW_COUNTER = 0

    # Initialize VGG16 models
    fire_model = keras.models.load_model('./vgg16/v2/NN_8_4_FireVGG16_Advanced.h5')
    smoke_model = keras.models.load_model('./vgg16/v2/NN_8_4_SmokeVGG16_Advanced.h5')

    # Cropped Detections storage
    list_of_cropped_detections = [] # for storing cropped detections
    list_of_coordinates_of_cropped_detections_fire = [] # for storing coordinates of cropped detections
    list_of_coordinates_of_cropped_detections_smoke = [] # for storing coordinates of cropped detections
    list_of_fire_confidence = [] # for storing confidence of cropped detections
    list_of_smoke_confidence = []

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:                          # webcam - also video from youtube etc.
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]


        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()


        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
      

        # Process detections
        for i, det in enumerate(pred):  
            if webcam: 
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            bounding_boxes_per_image = []
            
            # if(i==0) -- avoid having 1 frame multiple times in buffer
            if(i==0):
                IMG_BUFFER.put([im0, det, len(det)]) # len(det) gives information about how many detection there are, det passed for checking detection class (smoke or fire)

            if len(det): # if there is any detection
               # Rescale boxes from img_size to im0 size
               det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

               # Print results
               for c in det[:, -1].unique():
                  n = (det[:, -1] == c).sum()  # detections per class
                  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

               # Write results
               for *xyxy, conf, cls in reversed(det): # cls is the class (0 smoke or 1 fire)
                  bounding_boxes_per_image.append([int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])])
                  
                  # xyxy[1] = y1, xyxy[3] = y2, xyxy[0] = x1, xyxy[2] = x2
                  cropped_image = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                  list_of_cropped_detections.append([cropped_image, int(cls)])

                  # store confidences to calculate average
                  if(cls == 0): # smoke
                     list_of_smoke_confidence.append(round(float(conf), 2))
                  else: # fire
                     list_of_fire_confidence.append(round(float(conf), 2))


                  if(int(cls)==0):
                     list_of_coordinates_of_cropped_detections_smoke.append([int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])])
                  else:
                     list_of_coordinates_of_cropped_detections_fire.append([int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])])

                  if save_txt:  # Write to file
                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                     with open(txt_path + '.txt', 'a') as f:
                           f.write(('%g ' * len(line)).rstrip() % line + '\n')

                  if save_img or view_img:  # Add bbox to image
                     label = f'{names[int(cls)]} {conf:.2f}'
                     # comment under to mute bounding boxes
                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

               # If there is detection on a frame && IMG_BUFFER is full && IMG_BUFFER has atleast 3 frames with detection
               # -> then take snapshot of current buffer (ready_for_opt_flow : list) and send it for flow calculation
            else:
               if(frame % 30 == 0 and ("neg" in source)):
                  rand_x1 = random.randint(100, im0.shape[0])
                  rand_x2 = random.randint(100, im0.shape[0])
                  rand_y1 = random.randint(100, im0.shape[1])
                  rand_y2 = random.randint(100, im0.shape[1])

                  while((rand_x2<rand_x1) or ((rand_x2 - rand_x1) < 50)):
                     rand_x1 = random.randint(100, im0.shape[0])
                     rand_x2 = random.randint(100, im0.shape[0])
                  while((rand_y2<rand_y1) or ((rand_y2 - rand_y1) < 50)):
                     rand_y1 = random.randint(100, im0.shape[1])
                     rand_y2 = random.randint(100, im0.shape[1])
               
                  resized = cv2.resize(im0[rand_x1:rand_x2, rand_y1:rand_y2], (200, 200))
                  cv2.imwrite(f'./output_resized/background/background_{source[-9:-4]}_{frame}_smoke.jpg', resized)

            if(IMG_BUFFER.full()):
               # MAYBE define detections somewhere upwards and loop through IMG_BUFFER just in case there is enough detection to save process time
               # --------------   LOOP THROUGH IMG_BUFFER, END ON SAME VALUES ----------------
               ready_for_opt_flow = [] # TODO: this however causes buffer to reset after taking snapshop to 0 frames
               frames_with_detection = 0

               for i in range(IMG_BUFFER.qsize()): # Loops through IMG_BUFFER, ends on same values | just to fill ready_for_opt_flow list
                  current = IMG_BUFFER.get()
                  ready_for_opt_flow.append(current) # during loop we also create list of duplicate values

                  if(current[-1]): # current is list, first position is frame, second number of detections for that frame
                     frames_with_detection = frames_with_detection + 1

                  IMG_BUFFER.put(current)
               # -----------------------------------------------------------------------------

               # Check for >= 5 detections (in case BUFFER is size of 10), then take snapshots of buffer and detections,
               # then calculate average bounding box for reach class and draw flow from whole buffer, print it to last frame
               if(frames_with_detection >= math.floor(BUFFER_SIZE - BUFFER_SIZE/2)):
                  my_utils.snapshot_clear.clear_snapshot("./output/current_detection_snapshot")
                  my_utils.snapshot_clear.clear_snapshot("./output/current_buffer_average_bbox")
                  my_utils.snapshot_clear.clear_snapshot("./output/current_buffer_snapshot")
                  bbox_fire = []
                  bbox_smoke = []

                  # Taking snapshot of current buffer (-> saving frames as png's to folder)
                  index = 0
                  while not IMG_BUFFER.empty(): # empty magazine (IMG_BUFFER)
                        cv2.imwrite(f'./output/current_buffer_snapshot/{index}.jpg', IMG_BUFFER.get()[0])
                        index += 1

                  print("\n\n--------- BUFFER OUTPUT ---------")
                  # Taking snapshot of current detections (-> saving frames as png's to folder)
                  for index, detection in enumerate(list_of_cropped_detections):
                     if(detection[1] == 0): # 0 is smoke
                        cv2.imwrite(f'./output/current_detection_snapshot/smoke-{index}.jpg', detection[0])
                        bbox_smoke.append(detection[0])
                     else:
                        cv2.imwrite(f'./output/current_detection_snapshot/fire-{index}.jpg', detection[0])
                        bbox_fire.append(detection[0])

                  print("Number of detection in buffer: ", len(list_of_cropped_detections))
                  print("Number of smoke detections: ", len(bbox_smoke))
                  print("Number of fire detections: ", len(bbox_fire))
                  list_of_cropped_detections = [] # reset list of cropped detections

                  if(bbox_fire):
                     #print(list_of_coordinates_of_cropped_detections_fire)
                     print(list_of_coordinates_of_cropped_detections_fire)
                     average_bounding_box_fire = my_utils.bb_average.calculate_average_bbox(list_of_coordinates_of_cropped_detections_fire)
                     average_fire_detection = my_utils.bb_average.draw_average_bbox(average_bounding_box_fire, ready_for_opt_flow[-1][0], "fire")
                     list_of_coordinates_of_cropped_detections_fire.clear()
                     print("Average YOLOv7 confidence for fire: %.2f" % float(sum(list_of_fire_confidence)/len(list_of_fire_confidence)))
                     list_of_fire_confidence.clear()

                     resized = cv2.resize(average_fire_detection, (200, 200))
                     pred = fire_model.predict(resized.reshape(1, 200, 200, 3), verbose=0)

                     print("Fire prediction (from CNN): ", pred[0][0] * 100, "%")
                  
                     cv2.imwrite(f'./output/current_buffer_resized/fire.jpg', resized)
                     cv2.imwrite(f'./output_resized/fire/fire_{source[-9:-4]}_{frame}_fire.jpg', resized)

                        
                  if(bbox_smoke):
                     print("-----------------------")   
                     average_bounding_box_smoke = my_utils.bb_average.calculate_average_bbox(list_of_coordinates_of_cropped_detections_smoke)
                     average_smoke_detection = my_utils.bb_average.draw_average_bbox(average_bounding_box_smoke, ready_for_opt_flow[-1][0], "smoke")
                     list_of_coordinates_of_cropped_detections_smoke.clear()

                     print("Average YOLOv7 confidence for smoke: %.2f" % float(sum(list_of_smoke_confidence)/len(list_of_smoke_confidence)))
                     list_of_smoke_confidence.clear()

                     resized = cv2.resize(average_smoke_detection, (200, 200))
                     pred = smoke_model.predict(resized.reshape(1, 200, 200, 3), verbose=0)
                     
                     print("Smoke prediction (from CNN): ", pred[0][0] * 100, "%")
                     cv2.imwrite(f'./output/current_buffer_resized/smoke.jpg', resized)
                     cv2.imwrite(f'./output_resized/smoke/smoke_{source[-9:-4]}_{frame}_smoke.jpg', resized)


                  input("Press key to continue...")
                  print("--------- END BUFFER OUTPUT ---------\n")
            
            print("buffer size: ",IMG_BUFFER.qsize())

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
