import subprocess
import os
import cv2


def RunROC(f,s, confidnce):
   rootdir = './data/ROC-dataset/'
   label = None
   TP = 0
   FP = 0
   TN = 0
   FN = 0

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         print("in subdir: " + subdir)
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            
            label = subdir.split('/')[-1]
            if(label.__contains__("fire")):
               if(f):
                  print("FIRE - processing file: " + file)
                  label = 1
                  cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
                  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                  result = p.stdout.read().decode('utf-8').split("\n")
                  print(len(result[2:-5][0]))
                  if(len(result[2:-5][0]) -2 == 0):
                     FN += 1
                  else:
                     for res in result[2:-5][0]:
                        print("res: ",res)
                        if(res == "1" or res == "0"):
                           TP += 1
                           break
                  print("-------")
                  print("TP: ", TP)
                  print("FP: ", FP)
                  print("TN: ", TN)
                  print("FN: ", FN)
                  print("-------")
            elif(label.__contains__("smoke")):
               print("SMOKE - processing file: " + file)
               if(s):
                  label = 0
                  cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
                  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                  result = p.stdout.read().decode('utf-8').split("\n")
                  print(len(result[2:-5][0]))
                  if(len(result[2:-5][0]) -2 == 0):
                     FN += 1
                  else:
                     for res in result[2:-5][0]:
                        print("res: ",res)
                        if(res == "0" or res == "1"):
                           TP += 1
                           break
                  print("-------")
                  print("TP: ", TP)
                  print("FP: ", FP)
                  print("TN: ", TN)
                  print("FN: ", FN)
                  print("-------")
            else:
               print("BACKGROUND - processing file: " + file)
               label = None
               cmd = f"python3 ./detect.py --weights best.pt --conf {confidnce} --source {os.path.join(subdir, file)}"
               p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
               result = p.stdout.read().decode('utf-8').split("\n")
               print(len(result[2:-5][0]))
               if(len(result[2:-5][0]) -2 == 0):
                  TN += 1
               else:
                  for res in result[2:-5][0]:
                     print("res :",res)
                     if(res == "1" or res == "0"):
                        FP += 1
                        break
               print("-------")
               print("TP: ", TP)
               print("FP: ", FP)
               print("TN: ", TN)
               print("FN: ", FN)
               print("-------")

               
   return TP, FP, TN, FN


if __name__ == "__main__":
   TP, FP, TN, FN = RunROC(True, True, "0.05")
   print("TP: ", TP)
   print("FP: ", FP)
   print("TN: ", TN)
   print("FN: ", FN)
   

