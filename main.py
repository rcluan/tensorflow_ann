import argparse
import pandas as pd
from multilayerperceptron import MultiLayerPerceptron

parser = argparse.ArgumentParser()

"""
  Define the necessary parameters for running the neural network

  The number of epochs is not required
"""

parser.add_argument("-i", "--input", help="Number of inputs", type=int, required=True)
parser.add_argument("-hl", "--hidden", help="Number of hidden layers", type=int, required=True)
parser.add_argument("-o", "--output", help="Number of outputs", type=int, required=True)
parser.add_argument(
  "-n", "--neurons", 
  nargs="+", help="Number of Neurons for each hidden layer", 
  type=int, required=True
)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int)
parser.add_argument("-file", "--filename", help="Location of the file containing the data", type=str)
parser.add_argument("-sheet", "--sheetname", help="Name of the sheet on the data file", type=str)
parser.add_argument("-ext", "--extension", help="Data file extension (xlsx, csv, txt, xls)", type=str)

def main(args):

  try:
    # forces a non-zero and positive number of inputs
    if args.input <= 0:
      raise ValueError("The number of inputs cannot be zero or negative")
    # forces a non-zero and positive number of outputs
    if args.output <= 0:
      raise ValueError("The number of outputs cannot be zero or negative")
    # forces a non-zero and positive number of hidden layers
    if args.hidden <= 0:
      raise ValueError("The number of hidden layers cannot be zero or negative")
    # checks whether the length list passed as argument of the flag -n matches with the informed number of hidden layers
    if len(args.neurons) != args.hidden:
      raise ValueError("The number of hidden layers and the neuron's list length do not match")

    # forces a non-zero and positive number of neuros
    for n in args.neurons:
      if n <= 0:
        raise ValueError("The number of neurons cannot be zero or negative")

    filename = args.filename
    sheet_name = args.sheet_name

    data = pd.read_excel(filename, sheet_name)

    ann = MultiLayerPerceptron(args, data)
    ann.run()

  except ValueError as e:
    print 'Error: ' + e.message


if __name__ == "__main__":
  main(parser.parse_args())