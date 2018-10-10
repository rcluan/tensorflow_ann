import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot

np.set_printoptions(suppress=True)

DEFAULT_EPOCH = 50
DEFAULT_BATCH = 72

TOTAL = 744
N_TRAIN = 550

class KerasDenseMLP:

  def __init__(self, processor, args = None, values = []):

    self.processor = processor
    self.values = values

    self.hours = args.hours if args.hours else 1
    self.checkpoint = args.checkpoint if args.checkpoint else "dense_" + str(len(args.neurons)) + "_layers_checkpoint.keras"
    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH
    self.batch  = args.batch if args.batch else DEFAULT_BATCH

    self.build_data()
    self.model = keras.Sequential()
    
    """
        Adds an input layer containing the number of attributes specified along with the -i flag
      and having the hyperbolic tangent as the activation function.
        Considering a neuron list with the format [4,5], its output shape ought to be (*, 4).
    """
    self.model.add(keras.layers.Dense(
      units=args.neurons[0],
      activation="tanh",
      input_shape=(args.input,)
    ))

    """
        Adds hidden layers containing units according to the value supplied with the -n flag
      e.g. -n 4 5 yields a list [4,5] therefore two hidden layers will be added. The former
      having 4 neurons and the latter 5 neurons.
        Their output shapes ought to be respectively (*, 5) and (*, args.output).
    """
    for key, neurons in enumerate(args.neurons):
      if key < len(args.neurons) - 1:
        self.model.add(keras.layers.Dense(units=args.neurons[key+1], activation="tanh"))
      else:
        self.model.add(keras.layers.Dense(units=args.output, activation="tanh"))

    """
        Adds output input layer containing the number of attributes specified along with the -o flag
      and having the linear function as the activation function.
        Its output shape ought to be (*, args.output).
    """
    self.model.add(keras.layers.Dense(units=args.output))
    
    self.model.compile(
      optimizer=keras.optimizers.SGD(lr=args.learning),
      loss=keras.losses.MSE,
      metrics=[keras.metrics.MSE, keras.metrics.MAE]
    )

  def build_data(self):
    train = self.values[0:N_TRAIN]
    test = self.values[N_TRAIN:TOTAL]

    self.train_X, self.train_y = train[:, :-1], train[:, -1]
    self.test_X, self.test_y = test[:, :-1], test[:, -1]

  def train(self):
    print("train")
    checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint,
      monitor="loss",
      verbose=1,
      save_weights_only=True,
      save_best_only=True
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=0, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="./logs/", histogram_freq=0, write_graph=False)
    
    callbacks = [
      checkpoint, early_stopping, tensorboard
    ]
    
    history = self.model.fit(
      self.train_X, self.train_y, epochs=self.epochs,
      steps_per_epoch=36,
      validation_split=1,
      callbacks=callbacks
    )

    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()

  def evaluate(self):
    print("eval")
    result = self.model.evaluate(x=self.train_X, y=self.train_y)

    for res, metric in zip(result, self.model.metrics_names):
      print("{0}: {1:.3e}".format(metric, res))

  def predict(self):
    print("predict")

    y_reshaped, y_real = None, None

    for hour in range(self.hours):
      self.test_X, self.test_y = self.test_X[hour:], self.test_y[hour:]
      y_output = self.model.predict(self.test_X)

      self.test_X[:,4] = y_output.reshape(len(y_output))

      y_reshaped, y_real = self.processor.rescale(y_output, self.test_X, self.test_y)
    
    pyplot.plot(y_reshaped, label='predicted')
    pyplot.plot(y_real, label='measured')
    pyplot.legend()
    pyplot.show()