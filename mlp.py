import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot

np.set_printoptions(suppress=True)

DEFAULT_EPOCH = 50
DEFAULT_BATCH = 72

VALIDATION_SPLIT = 0.2
TESTING_SPLIT = 0.3

TOTAL = 744
N_INPUTS = 550

class KerasDenseMLP:

  def __init__(self, processor, args = None, values = []):

    self.processor = processor
    self.values = values

    self.hours = args.hours if args.hours else 1
    self.checkpoint = args.checkpoint if args.checkpoint else "dense_" + str(len(args.neurons)) + "_layers_checkpoint.keras"
    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH
    self.batch  = args.batch if args.batch else DEFAULT_BATCH

    self.features = args.input

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
    inputs = self.values[0:N_INPUTS]
    prediction = self.values[N_INPUTS:TOTAL]
    
    np.random.shuffle(inputs)
    N_TEST = int(N_INPUTS*TESTING_SPLIT)

    self.train_X, self.train_y = inputs[:(N_INPUTS-N_TEST), :-1], inputs[:(N_INPUTS-N_TEST), -1]
    self.test_X, self.test_y = inputs[(N_INPUTS-N_TEST):, :-1], inputs[(N_INPUTS-N_TEST):, -1]

    self.prediction_X, self.prediction_y = prediction[:, :-1], prediction[:, -1]



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
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        min_lr=1e-4,
        patience=0,
        verbose=1
    )
    
    callbacks = [
      checkpoint, early_stopping, tensorboard, reduce_lr
    ]
    
    history = self.model.fit(
      self.train_X, self.train_y, epochs=self.epochs,
      steps_per_epoch=72,
      validation_steps=72,
      validation_split=VALIDATION_SPLIT,
      callbacks=callbacks
    )

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

  def evaluate(self):
    print("eval")
    result = self.model.evaluate(x=self.test_X, y=self.test_y)

    for res, metric in zip(result, self.model.metrics_names):
      print("{0}: {1:.3e}".format(metric, res))

  def predict(self):
    print("predict")

    y_reshaped, y_real = None, self.processor.rescale(self.prediction_y.reshape(len(self.prediction_y), 1), self.prediction_X)

    for hour in range(self.hours):
      self.prediction_X = self.prediction_X[1:]
      y_output = self.model.predict(self.prediction_X)

      self.prediction_X[:,4] = y_output.reshape(len(y_output))

      y_reshaped = self.processor.rescale(y_output, self.prediction_X)
    

    y = np.zeros(y_real.shape)
    np.put(y, np.indices(y.shape), np.nan)
    starting_index = len(y) - len(y_reshaped)

    np.put(y, np.indices(y.shape)[:,starting_index:],y_reshaped)

    pyplot.plot(y, label='predicted')
    pyplot.plot(y_real, label='measured')
    pyplot.legend()
    pyplot.show()