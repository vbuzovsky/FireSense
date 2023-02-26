from .utils import load_flow_dataset, shuffle_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def Train_and_Load_Model():
   raw_data, raw_labels = load_flow_dataset()
   data, labels = shuffle_dataset(raw_data, raw_labels)
   
   x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
   x_train_partial, x_val, y_train_partial, y_val = train_test_split(x_train, y_train, test_size=0.2)

   model = keras.Sequential([
      layers.Dense(800, activation="relu", name="layer1"),
      layers.Dense(600, activation="relu", name="layer2"),
      layers.Dense(400, activation="relu", name="layer3"),
      layers.Dense(200, activation="relu", name="layer4"),
      layers.Flatten(),
      layers.Dense(1, activation="sigmoid", name="layer5")
   ])

   model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])


   y_train_partial = tf.one_hot(y_train_partial, depth=1)
   y_val = tf.one_hot(y_val, depth=1)
   
   history = model.fit(np.array(x_train_partial), y_train_partial, epochs=6, batch_size=32, validation_data=(np.array(x_val), y_val))

   history_dict = history.history
   loss_values = history_dict["loss"]
   val_loss_values = history_dict["val_loss"]
   epochs = range(1, len(loss_values) + 1)
   plt.plot(epochs, loss_values, "bo", label="Training loss")
   plt.plot(epochs, val_loss_values, "b", label="Validation loss")
   plt.title("Training and validation loss")
   plt.xlabel("Epochs")
   plt.ylabel("Loss")
   plt.legend()
   plt.show()
   



if(__name__ == "__main__"):
   Train_and_Load_Model()