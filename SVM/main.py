from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
import os

def load_optical_flow(path):
    flow = np.load(path, allow_pickle=True, fix_imports=True)
    return flow

def save_optical_flow(flow, path):
   try:
      np.save(path, flow)
      return True
   except:
      return False


def load_flow_dataset():
   rootdir = './data/FlowDataset'
   data = []
   labels = []

   for subdir, dirs, files in os.walk(rootdir):
      for file in files:
         if(not file.__contains__(".DS_Store")): # Ignore .DS_Store files in MacOS      
            data.append(load_optical_flow(os.path.join(subdir, file)))

            label = subdir.split('/')[-1]
            if(label.__contains__("negative")):
               labels.append(0)
            else:
               labels.append(1)

   return data, labels

# this is copilot code -- POSSIBLY NOT WORKING CORRECTLY
def shuffle_dataset(data, labels):
   combined = list(zip(data, labels))
   np.random.shuffle(combined)
   data[:], labels[:] = zip(*combined)

   return data, labels

def Train_and_Load_Model():
   raw_data, raw_labels = load_flow_dataset()
   data, labels = shuffle_dataset(raw_data, raw_labels)
   
   x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
   print("\nInitialising SVM model...")
   model = svm.SVC()

   print("Training SVM model...")
   x_train = np.array(x_train)
   model.fit(x_train.reshape(x_train.shape[0], 800), y_train)
   
   # Print achieved accuracy
   y_pred = model.predict(np.array(x_test).reshape(np.array(x_test).shape[0], 800))
   print('Achieved accuracy with current flowdataset: %.3f' % accuracy_score(y_test, y_pred))
   print('Achieved precision with current flowdataset: %.3f' % average_precision_score(y_test, y_pred))
   print("\n----------------------")

   return model, accuracy_score(y_test, y_pred), average_precision_score(y_test, y_pred)

def pred(model, flow):
   return model.predict(np.array(flow).reshape(1, 800))
   
if __name__ == '__main__':
   Train_and_Load_Model()