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
   number_of_positive = 0
   number_of_negative = 0

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            
            label = subdir.split('/')[-1]
            if(label.__contains__("fire") and cls == "fire"):
               data.append(_load_optical_flow(os.path.join(subdir, file)))
               if(label.__contains__("negative")):
                  labels.append(0)
                  number_of_negative += 1
               else:
                  labels.append(1)
                  number_of_positive += 1
            elif(label.__contains__("smoke") and cls == "smoke"):
               data.append(_load_optical_flow(os.path.join(subdir, file)))
               if(label.__contains__("negative")):
                  labels.append(0)
                  number_of_negative += 1
               else:
                  labels.append(1)
                  number_of_positive += 1

   print(len(data), len(labels))
   print(number_of_positive, number_of_negative)
   return data, labels

# this is copilot code -- POSSIBLY NOT WORKING CORRECTLY
def shuffle_dataset(data, labels):
   combined = list(zip(data, labels))
   np.random.shuffle(combined)
   data[:], labels[:] = zip(*combined)

   return data, labels

