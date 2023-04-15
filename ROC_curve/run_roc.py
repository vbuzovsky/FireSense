import subprocess
import os
import cv2


def RunROC(f,s, confidnce):
   rootdir = './data/ROC-test/'
   labels = []

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         print("in subdir: " + subdir)
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            
            label = subdir.split('/')[-1]
            if(label.__contains__("fire.jpg")):
               if(f):
                  labels.append(1)
                  cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
                  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                  result = p.stdout.read().decode('utf-8').split("\n")
                  print(result[2:-5])
                  print("number of detections: ", len(result[2:-5]))

            elif(label.__contains__("smoke.jpg")):
               if(s):
                  labels.append(1)
                  cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
                  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                  result = p.stdout.read().decode('utf-8').split("\n")
                  print(result[2:-5])
                  print("number of detections: ", len(result[2:-5]))

            else:
               labels.append(0)
               cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
               p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
               result = p.stdout.read().decode('utf-8').split("\n")
               print(result[2:-5])
               print("number of detections: ", len(result[2:-5]))

   return data, labels


if __name__ == "__main__":
   data, labels = RunROC(True, False, "0.25")
   print(len(data))
   print(len(labels))

