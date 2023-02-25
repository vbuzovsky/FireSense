import file_manager as fm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

def main():
   raw_data, raw_labels = fm.load_flow_dataset()
   data, labels = fm.shuffle_dataset(raw_data, raw_labels)
   
   x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
   model = svm.SVC()

   print("x_train: ", np.array(x_train).shape)
   x_train = np.array(x_train)
   model.fit(x_train.reshape(x_train.shape[0], 800), y_train)
   
   y_pred = model.predict(np.array(x_test).reshape(np.array(x_test).shape[0], 800))
   print("y_test: ", y_test[0])
   print("\npredictions: ", y_pred[0])
   #print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))



if __name__ == '__main__':
   main()