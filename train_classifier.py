import Classifiers.keras_nn as NN
import argparse

def main(cls):
    NN.Train_and_Load_Model(cls)

if(__name__ == "__main__"):
   parser = argparse.ArgumentParser()
   parser.add_argument('--cls', type=str, help='select the class to train the model')
   parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
   args = parser.parse_args()
   NN.Train_and_Load_Model(args.cls, args.epochs)
