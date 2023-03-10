import numpy as np
import math
import cv2

def subsample(flow):
   subsampled_flow = []

   # print(flow.shape)
   # print(flow.shape[0])
   # print(flow.shape[1])

   x = flow.shape[0]
   y = flow.shape[1]

   subsampled = cv2.resize(flow, (20,20))
   print(subsampled.shape)
   return subsampled

   # for flows that are smaller than 20x20x2
   if(x < 20):
      x = 20
   if(y < 20):
      y = 20


   subsampled_rows = flow[0::math.floor(x/20)]
   for row in subsampled_rows:
      subsampled_points = row[0::math.floor(y/20)]
      subsampled_flow.append(subsampled_points)

   #print(np.array(subsampled_flow)[:20, :20, :])
   return np.array(subsampled_flow)[:20, :20, :]
      



