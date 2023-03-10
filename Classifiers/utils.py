import numpy as np
import os



def _load_optical_flow(path):
    flow = np.load(path, allow_pickle=True, fix_imports=True)
    return flow

def _save_optical_flow(flow, path):
   try:
      np.save(path, flow)
      return True
   except:
      return False

# OLD BOTH CLASSES DATASET LOADER
# def load_flow_dataset():
#    rootdir = './data/NEW_FLOW'
#    data = []
#    labels = []

#    for subdir, dirs, files in os.walk(rootdir):
#       for file in files:
#          if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
#             data.append(_load_optical_flow(os.path.join(subdir, file)))

#             label = subdir.split('/')[-1]
#             if(label.__contains__("negative")):
#                labels.append(0)
#             else:
#                labels.append(1)

#    return data, labels

def load_flow_dataset(cls):
   rootdir = './data/FINAL_DATASET'
   data = []
   labels = []

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            
            label = subdir.split('/')[-1]
            if(label.__contains__("fire") and cls == "fire"):
               data.append(_load_optical_flow(os.path.join(subdir, file)))
               if(label.__contains__("negative")):
                  labels.append(0)
               else:
                  labels.append(1)
            elif(label.__contains__("smoke") and cls == "smoke"):
               data.append(_load_optical_flow(os.path.join(subdir, file)))
               if(label.__contains__("negative")):
                  labels.append(0)
               else:
                  labels.append(1)

   return data, labels

# just checking dataset properties
def calc_average_flow(flow):
   x = []
   y = []
   for i in range(0, flow.shape[0]):
      for j in range(0, flow.shape[1]):
         x.append(flow[i, j, :][0])
         y.append(flow[i, j, :][1])
   
   # print("x: ", np.mean(x))
   # print("y: ", np.mean(y))
   return [np.mean(x), np.mean(y)]

def shuffle_dataset(data, labels):
   combined = list(zip(data, labels))
   np.random.shuffle(combined)
   data[:], labels[:] = zip(*combined)

   return data, labels




if __name__ == '__main__':
   raw_data, raw_labels = load_flow_dataset("fire")

   average_positive = []
   average_negative = []
   for index, item in enumerate(raw_data):
      if(raw_labels[index] == 1):
         average_positive.append(calc_average_flow(item))
      else:
         average_negative.append(calc_average_flow(item))
   
   total_mean_x = []
   total_mean_y = []
   for average_flow in average_positive:
      total_mean_x.append(average_flow[0])
      total_mean_y.append(average_flow[1])
   
   print("positive average: [", np.mean(total_mean_x), np.mean(total_mean_y), "]")
   
   total_mean_x = []
   total_mean_y = []

   for average_flow in average_negative:
      total_mean_x.append(average_flow[0])
      total_mean_y.append(average_flow[1])

   print("negative average: [", np.mean(total_mean_x), np.mean(total_mean_y), "]")


