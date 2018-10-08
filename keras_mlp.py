import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot

np.set_printoptions(suppress=True)

DEFAULT_EPOCH = 50
DEFAULT_BATCH = 72

TOTAL = 744
N_TRAIN = 550

class KerasMLP:

  def __init__(self, args = None, values = []):

    train = values[:N_TRAIN]
    test = values[N_TRAIN:TOTAL]

    self.checkpoint = "checkpoint.keras"

    self.train_X, self.train_y = train[:, :-1], train[:, -1]
    self.test_X, self.test_y = test[:, :-1], test[:, -1]

    self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
    self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
    
    print(self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH
    self.batch  = args.batch if args.batch else DEFAULT_BATCH

    self.model = keras.Sequential()
    
    """
        Adds an input layer containing the number of attributes specified along with the -i flag
      and having the hyperbolic tangent as the activation function.
        Considering a neuron list with the format [4,5], its output shape ought to be (*, 4).
    """
    self.model.add(keras.layers.Dense(
      units=args.neurons[0],
      activation="tanh",
      input_shape=(self.train_X.shape[1], self.train_X.shape[2])
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

  def train(self):
    print("train")
    checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint,
      monitor="val_loss",
      verbose=1,
      save_weights_only=True,
      save_best_only=True
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="./logs/", histogram_freq=0, write_graph=False)
    
    callbacks = [
      checkpoint, early_stopping, tensorboard
    ]
    
    history = self.model.fit(
      self.train_X, self.train_y, epochs=self.epochs, batch_size=self.batch,
      steps_per_epoch=75,
      validation_split=1,
      callbacks=callbacks
    )

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

  def evaluate(self):
    print("eval")
    result = self.model.evaluate(x=np.expand_dims(self.train_X, axis=0), y=np.expand_dims(self.train_y, axis=0))

    if False:
      for res, metric in zip(result, self.model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))

  def predict(self):
    print("predict")
    y_output = self.model.predict(self.test_X)

    print (y_output)
    




