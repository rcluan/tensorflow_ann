import argparse
import pandas as pd
from mlp import KerasDenseMLP
from data import DataProcessor

parser = argparse.ArgumentParser()

"""
  Define the necessary parameters for running the neural network

  The number of epochs is not required
"""

parser.add_argument(
  "-n", "--neurons", 
  nargs="+", help="Number of Neurons for each Dense hidden layer", type=int
)
parser.add_argument("-lr", "--learning", help="Learning Constant", type=float)

parser.add_argument("-hours", "--hours", help="Number of hours to consider for prediction", type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int)
parser.add_argument("-b", "--batch", help="Batch size", type=int)

def main(args):

  try:

    hours = args.hours if args.hours else 1

    filename = "dataset.csv"
    dataset = pd.read_csv(filename, header=0, index_col=0)

    processor = DataProcessor(dataset)
    processor.shift(hours)

    x_data, y_data = processor.to_numpy_arrays(hours)
    processor.build_dataset(x_data, y_data)

    mld = KerasDenseMLP(processor=processor,args=args)

    try:
      mld.model.load_weights(mld.checkpoint)
        
      mld.evaluate()
      mld.predict()
    except Exception as error:
      print("Error trying to load checkpoint.")
      print(error)
      mld.train()
      mld.evaluate()
      mld.predict()


  except ValueError as e:
    print ('Error: ' + e.message)


if __name__ == "__main__":
  main(parser.parse_args())