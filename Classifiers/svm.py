from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np

from .utils import load_flow_dataset, shuffle_dataset


def Train_and_Load_Model(cls):
   raw_data, raw_labels = load_flow_dataset(cls)
   data, labels = shuffle_dataset(raw_data, raw_labels)
   
   x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
   print(f"\nInitialising SVM {cls} model...")
   model = svm.SVC()

   print(f"Training SVM {cls} model...")
   x_train = np.array(x_train)
   model.fit(x_train.reshape(x_train.shape[0], 800), y_train)
   
   # Print achieved accuracy
   y_pred = model.predict(np.array(x_test).reshape(np.array(x_test).shape[0], 800))
   print(f'Achieved accuracy ({cls} model): %.3f' % accuracy_score(y_test, y_pred))
   print(f'Achieved precision ({cls} model): %.3f' % average_precision_score(y_test, y_pred))
   print("\n----------------------")

   return model, accuracy_score(y_test, y_pred), average_precision_score(y_test, y_pred)

def pred(model, flow):
   return model.predict(np.array(flow).reshape(1, 800))
   
if __name__ == '__main__':
   Train_and_Load_Model()