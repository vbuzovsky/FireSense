import copy
import cv2

# bboxes: [ [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])], ... ]
# [y1, y2, x1, x2]
def calculate_average_bbox(bboxes : list):
   y1_list = []
   y2_list = []
   x1_list = []
   x2_list = []
   
   for i in range(len(bboxes)):
         y1_list.append(bboxes[i][0])
         y2_list.append(bboxes[i][1])
         x1_list.append(bboxes[i][2])
         x2_list.append(bboxes[i][3])


   return [min(y1_list), max(y2_list), min(x1_list), max(x2_list)]

       
def draw_average_bbox(bbox : list, img, cls):
   cropped_img_with_average_bounding_box = img[bbox[0]:bbox[1],bbox[2]:bbox[3]]
   cv2.imwrite(f'./output/current_buffer_average_bbox/{cls}.jpg', cropped_img_with_average_bounding_box)
   return cropped_img_with_average_bounding_box
