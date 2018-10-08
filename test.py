from data import DataProcessor
import pandas as pd
from keras_mlp import KerasMLP

filename = "dataset.csv"
dataset = pd.read_csv(filename, header=0, index_col=0)

processor = DataProcessor(dataset)

processor.scale()
reframed = processor.series_to_supervised(1, 1)
reframed.drop(reframed.columns[[9,10,11,12,14,15,16,17]], axis=1, inplace=True)

mld = KerasMLP(values=reframed.values)

#print (reframed.head(5))