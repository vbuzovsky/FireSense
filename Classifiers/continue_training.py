from utils import shuffle_dataset, LoadDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D



def Train_and_Load_Model(epochs):
   x, y = shuffle_dataset(*LoadDataset())
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

   loaded_model = tf.keras.models.load_model('NN_28_3_classificator.h5')

   history = model.fit(np.array(x_train), y_train, epochs=epochs, batch_size=32, validation_data=(np.array(x_val), y_val) , callbacks=callbacks)
   

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
   plt.savefig("continue_training.png")
   
   model.save(f"NN_28_3_classificator_continue.h5")



if(__name__ == "__main__"):
   Train_and_Load_Model(100)