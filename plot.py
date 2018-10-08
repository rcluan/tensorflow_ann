# based on https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import pandas as pd
from data import DataProcessor
from matplotlib import pyplot

filename = 'dataset.csv'

dataset = pd.read_csv(filename, header=0, index_col=0)
values = dataset.values

groups = range(9)
i = 1

pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right'),
	i+=1
	
pyplot.show()