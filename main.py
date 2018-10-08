import argparse
import pandas as pd
from mlp import MultiLayerPerceptron
from keras_mlp import KerasMLP
from data import DataProcessor

parser = argparse.ArgumentParser()

"""
  Define the necessary parameters for running the neural network

  The number of epochs is not required
"""

parser.add_argument("-i", "--input", help="Number of inputs", type=int, required=True)
parser.add_argument("-o", "--output", help="Number of outputs", type=int, required=True)
parser.add_argument(
  "-n", "--neurons", 
  nargs="+", help="Number of Neurons for each hidden layer", 
  type=int, required=True
)
parser.add_argument("-lc", "--learning", help="Learning Constant", type=float, required=True)

parser.add_argument("-e", "--epochs", help="Number of epochs", type=int)
parser.add_argument("-b", "--batch", help="Batch size", type=int)

def main(args):

  try:
    # forces a non-zero and positive number of inputs
    if args.input <= 0:
      raise ValueError("The number of inputs cannot be zero or negative")
    # forces a non-zero and positive number of outputs
    if args.output <= 0:
      raise ValueError("The number of outputs cannot be zero or negative")

    # forces a non-zero and positive number of neuros
    for n in args.neurons:
      if n <= 0:
        raise ValueError("The number of neurons cannot be zero or negative")

    #data_processor = DataProcessor(filename, sheet_name)

    #ann = KerasMLP(args, data_processor.processed_data)
    #ann.run()
    #ann = MultiLayerPerceptron(args, data_processor)
    #ann.run()

  except ValueError as e:
    print 'Error: ' + e.message


if __name__ == "__main__":
  main(parser.parse_args())