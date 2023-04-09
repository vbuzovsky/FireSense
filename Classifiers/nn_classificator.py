from utils import shuffle_dataset, LoadDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D




def Train_and_Load_Model(epochs):
   x, y = shuffle_dataset(*LoadDataset(f=True,s=False))
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


   # Load pre-trained MobileNetV2 model, without the top layer
   base_model = VGG16(input_shape=(200, 200, 3), include_top=False)

   # Freeze the pre-trained layers
   for layer in base_model.layers:
      layer.trainable = False

   # Add a new top layer for classification
   model = Sequential()
   model.add(base_model)
   model.add(GlobalAveragePooling2D())
   model.add(Dense(3, activation='softmax'))

   model.summary()

   opt = keras.optimizers.RMSprop(learning_rate=0.0005)
   model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

   y_train = tf.one_hot(y_train, depth=3)
   y_val = tf.one_hot(y_val, depth=3)
   
   callbacks = [
      keras.callbacks.EarlyStopping(
         monitor="val_loss",
         restore_best_weights=True,
         patience=10,
      )
   ]

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
   plt.savefig("VGG16.png")
   
   model.save(f"NN_6_4_classificator.h5")



if(__name__ == "__main__"):
   Train_and_Load_Model(200)