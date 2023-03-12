from utils import shuffle_dataset, LoadDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def Train_and_Load_Model(epochs):
   x, y = shuffle_dataset(*LoadDataset())
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

   model = keras.Sequential([
      layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200, 200, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(3, activation="sigmoid")
   ])

   opt = keras.optimizers.RMSprop(learning_rate=0.00001)
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
   plt.show()
   
   model.save(f"NN_classificator.h5")



if(__name__ == "__main__"):
   Train_and_Load_Model(20)