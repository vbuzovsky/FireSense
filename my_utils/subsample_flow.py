import numpy as np
import math

def subsample(flow):
   subsampled_flow = []

   x = flow.shape[0]
   y = flow.shape[1]

   print("x: ", x)
   print("y: ", y)

   subsampled_rows = flow[0::math.floor(x/20)]
   for row in subsampled_rows:
      subsampled_points = row[0::math.floor(y/20)]
      subsampled_flow.append(subsampled_points)

   # print("subsampled_flow: ", np.array(subsampled_flow).shape)
   # print("subsampled_flow: ", np.array(subsampled_flow))

   return np.array(subsampled_flow)[:20, :20, :]
      



