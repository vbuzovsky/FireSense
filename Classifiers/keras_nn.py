from .utils import load_flow_dataset, shuffle_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model


def Train_and_Load_Model(cls, epochs):
   data, labels = load_flow_dataset(cls)
   #data, labels = shuffle_dataset(raw_data, raw_labels)
   
   x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

   model = keras.Sequential([
      layers.Dense(800, activation="relu", name="layer1"),
      layers.Dense(400, activation="relu", name="layer2"),
      layers.Dense(200, activation="relu", name="layer3"),
      layers.Dense(100, activation="relu", name="layer4"),
      layers.Flatten(),
      layers.Dense(1, activation="sigmoid", name="layer5")
   ])

   opt = keras.optimizers.RMSprop(learning_rate=0.0001)
   model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

   # model.build(input_shape=(None, 20, 20, 2))
   # #plot_model(model, to_file=f"NN_{cls}3_classificator.png", show_shapes=True, show_layer_names=True)
   # print(model.summary())
   
   print("y_train before", y_train)
   y_train = tf.one_hot(y_train, depth=1)
   print("y_train after", y_train)
   y_val = tf.one_hot(y_val, depth=1)
   
   callbacks = [
      keras.callbacks.EarlyStopping(
         monitor="val_loss",
         restore_best_weights=True,
         patience=10,
      )
   ]

   history = model.fit(np.array(x_train), y_train, epochs=epochs, batch_size=16, validation_data=(np.array(x_val), y_val) , callbacks=callbacks)

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
   
   model.save(f"NN_{cls}Test_classificator.h5")



if(__name__ == "__main__"):
   Train_and_Load_Model()