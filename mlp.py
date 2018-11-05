import tensorflow as tf
from tensorflow import keras
from tensorflow import set_random_seed

from tensorflow.keras import backend as K

import numpy as np
from numpy.random import seed

from matplotlib import pyplot as plt

import ai_metrics as metrics

np.set_printoptions(suppress=True)
seed(1)
set_random_seed(2)

DEFAULT_EPOCH = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_HOURS = 1
DEFAULT_NEURONS = [9,6]
DEFAULT_LR = 1e-2

class KerasDenseMLP:

  def __init__(self, processor, args = None):

    self.processor = processor

    self.hours = args.hours if args.hours else DEFAULT_HOURS
    self.batch  = args.batch if args.batch else DEFAULT_BATCH_SIZE
    self.neurons = args.neurons if args.neurons else DEFAULT_NEURONS
    self.learning = args.learning if args.learning else DEFAULT_LR
    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH

    self.checkpoint = "mlp_" + str(len(self.neurons)) + "_layers_" + str(self.hours) + "_hours_checkpoint.keras"

    #self.processor.generate_validation_data()
    self.model = keras.Sequential()
    
    """
        Adds an input layer containing the number of attributes specified along with the -i flag
      and having the hyperbolic tangent as the activation function.
        Considering a neuron list with the format [4,5], its output shape ought to be (*, 4).
    """
    self.model.add(keras.layers.Dense(
      units=self.neurons[0],
      activation="tanh",
      input_shape=(self.processor.n_inputs,)
    ))

    """
        Adds hidden layers containing units according to the value supplied with the -n flag
      e.g. -n 4 5 yields a list [4,5] therefore two hidden layers will be added. The former
      having 4 neurons and the latter 5 neurons.
        Their output shapes ought to be respectively (*, 5) and (*, args.output).
    """
    for key, neurons in enumerate(self.neurons):
      if key < len(self.neurons) - 1:
        self.model.add(keras.layers.Dense(units=self.neurons[key+1], activation="tanh"))
      else:
        self.model.add(keras.layers.Dense(units=self.processor.n_outputs, activation="tanh"))

    """
        Adds output input layer containing the number of attributes specified along with the -o flag
      and having the linear function as the activation function.
        Its output shape ought to be (*, args.output).
    """
    self.model.add(keras.layers.Dense(units=self.processor.n_outputs))
    
    print(self.model.summary())
    self.model.compile(
      optimizer=keras.optimizers.RMSprop(lr=self.learning),
      loss=keras.losses.MSE,
      metrics=[
        keras.metrics.MAE
      ]
    )

  def train(self):
    checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint,
      monitor="val_loss",
      verbose=1,
      save_weights_only=True,
      save_best_only=True
    )
    early_stopping = keras.callbacks.EarlyStopping(
      monitor="val_loss",
      patience=10,
      min_delta=1e-4,
      mode='auto',
      verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        min_lr=1e-4,
        patience=0,
        verbose=1
    )
    
    callbacks = [
      checkpoint, early_stopping, reduce_lr
    ]
    
    history = self.model.fit(
      x=self.processor.x_train_scaled,
      y=self.processor.y_train,
      epochs=self.epochs,
      batch_size=self.batch,
      validation_split=self.processor.validation_split
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.savefig("loss.png")

  def evaluate(self):
    result = self.model.evaluate(x=self.processor.x_test_scaled, y=self.processor.y_test)

    for res, metric in zip(result, self.model.metrics_names):
      print("{0}: {1:.3e}".format(metric, res))

  def predict(self):
    x = self.processor.x_test_scaled
    y = self.processor.y_test

    y_test = self.model.predict(x)

    # For each output-signal.
    for signal in range(len(self.processor.target_names)):
      # Get the output-signal predicted by the model.
      signal_pred = y_test[:, signal]
        
      # Get the true output-signal from the data-set.
      signal_true = y[:, signal]
      
      plt.clf()
      # Plot and compare the two signals.
      plt.plot(signal_true, label='Measured')
      plt.plot(signal_pred, label='Prediction')
        
      # Plot labels etc.
      plt.ylabel(self.processor.target_names[signal])
      plt.legend()
      plt.savefig("prediction_"+self.processor.target_names[signal]+".png")
    
    """
    def predict(self):
      print("predict")

      y_reshaped, y_real = None, self.processor.rescale(self.prediction_X[:,4].reshape(len(self.prediction_X), 1), self.prediction_X)

      for hour in range(self.hours):
        self.prediction_X = self.prediction_X[1:] if hour > 1 else self.prediction_X[0:]
        y_output = self.model.predict(self.prediction_X)

        self.prediction_X[:,4] = y_output.reshape(len(y_output))

        y_reshaped = self.processor.rescale(y_output, self.prediction_X)
      
      index_ = self.hours-2 if self.hours > 1 else 0
      rmse = sqrt(mean_squared_error(y_reshaped, y_real[index_:]))
      print('RMSE: %.3f' % rmse)

      y = np.zeros(y_real.shape)
      np.put(y, np.indices(y.shape), np.nan)
      starting_index = len(y) - len(y_reshaped)

      np.put(y, np.indices(y.shape)[:,starting_index:],y_reshaped)

      plt.plot(y, label='predicted')
      plt.plot(y_real, label='measured')
      plt.legend()
      plt.show()
    """