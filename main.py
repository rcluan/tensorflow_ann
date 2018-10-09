import argparse
import pandas as pd
from dense_mlp import KerasDenseMLP
from gru_mlp import KerasGRUMLP
from lstm_mlp import KerasLSTMMLP
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
  nargs="+", help="Number of Neurons for each Dense hidden layer", type=int
)
parser.add_argument("-lc", "--learning", help="Learning Constant", type=float, required=True)

parser.add_argument("-hours", "--hours", help="Number of hours to consider for prediction", type=int)
parser.add_argument("-cp", "--checkpoint", help="Checkpoint file", type=int)
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

    hours = args.hours if args.hours else 1
    
    filename = "dataset.csv"
    dataset = pd.read_csv(filename, header=0, index_col=0)

    processor = DataProcessor(dataset)

    processor.scale()
    reframed = processor.series_to_supervised(1, 1)
    reframed.drop(reframed.columns[[9,10,11,12,14,15,16,17]], axis=1, inplace=True)

    print("Select layer type")
    print("1 - Dense")
    print("2 - GRU")
    print("3 - LSTM")

    type = input()

    if(type == "1"):
      mld = KerasDenseMLP(processor=processor,args=args, values=reframed.values)
    elif(type == "2"):
      mld = KerasGRUMLP(processor=processor,args=args, values=reframed.values)
    elif(type == "3"):
      mld = KerasLSTMMLP(processor=processor,args=args, values=reframed.values)
    else:
      raise ValueError("Invalid layer")

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