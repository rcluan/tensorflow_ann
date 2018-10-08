from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

TRAINING_PERCENTAGE = 0.7392

class DataProcessor:

  def __init__(self, dataset):
    self.dataset = dataset
    self.encoder = LabelEncoder()
    self.scaler = MinMaxScaler()

  def scale(self):

    values = self.dataset.values
    values[:,4] = self.encoder.fit_transform(values[:,4])
    values = values.astype('float32')

    self.scaled_dataset = self.scaler.fit_transform(values)
    
  # from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
  # convert series to supervised learning
  def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(self.scaled_dataset) is list else self.scaled_dataset.shape[1]
    df = DataFrame(self.scaled_dataset)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    return agg

    return self.scaled_dataset

  def rescale(self, y_output):

    print (y_output)
    
    y_output_rescaled = self.scaler.inverse_transform(y_output)

    return y_output_rescaled
