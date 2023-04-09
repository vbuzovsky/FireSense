import os
import cv2
import numpy as np

def _load_img(path):
    img = cv2.imread(path)
    return img

def LoadDataset(f,s):
   rootdir = './data/CNN_dataset/'
   data = []
   labels = []

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            
            label = subdir.split('/')[-1]
            if(label.__contains__("fire")):
               if(f):
                  data.append(_load_img(os.path.join(subdir, file)))
                  labels.append(0)
            elif(label.__contains__("smoke")):
               if(s):
                  data.append(_load_img(os.path.join(subdir, file)))
                  labels.append(1)
            else:
               data.append(_load_img(os.path.join(subdir, file)))
               labels.append(2)

   return data, labels
    
def shuffle_dataset(data, labels):
   combined = list(zip(data, labels))
   np.random.shuffle(combined)
   data[:], labels[:] = zip(*combined)

   return data, labels

if __name__ == "__main__":
   data, labels = shuffle_dataset(*LoadDataset())
   print(len(data))
   print(len(labels))
   print(data[0].shape)
   cv2.imshow("img", data[0])
   print(labels[:10])
   cv2.waitKey(0)
