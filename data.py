from math import sqrt
from numpy import concatenate
from numpy import delete
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

TARGET_NAMES = ['v_anemo']

NUM_TOTAL = 744
NUM_TRAIN = 550

VALIDATION_SPLIT = 0.2

class DataProcessor:

  def __init__(self, dataset):

    dataset.values.astype('float32')

    self.raw_dataset = dataset
    self.encoder = LabelEncoder()
    
    self.x_scaler = MinMaxScaler((-1,1))
    self.y_scaler = MinMaxScaler((-1,1))

    self.target_names = TARGET_NAMES
    self.validation_split = VALIDATION_SPLIT

  def shift(self, shift_steps):
    self.dataset_targets = self.raw_dataset[TARGET_NAMES].shift(-shift_steps)

  def to_numpy_arrays(self, shift_steps):
    x_data = self.raw_dataset.values[0:-shift_steps]
    y_data = self.dataset_targets.values[:-shift_steps]
  
    return x_data, y_data

  def build_dataset(self, x_data, y_data):

    self.n_inputs = x_data.shape[1]
    self.n_outputs = y_data.shape[1]

    self.x_train = x_data[:NUM_TRAIN]
    self.y_train = y_data[:NUM_TRAIN]

    self.x_test = x_data[NUM_TRAIN:NUM_TOTAL]
    self.y_test = y_data[NUM_TRAIN:NUM_TOTAL]

    self.x_train_scaled = self.x_scaler.fit_transform(self.x_train)
    #self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
    
    self.x_test_scaled = self.x_scaler.transform(self.x_test)
    #self.y_test_scaled = self.y_scaler.transform(self.y_test)

  def rescale(self, data, x=True):
    return self.x_scaler.inverse_transform(data[0]) if x else self.y_scaler.inverse_transform(data[0])