import tensorflow as tf
from tensorflow import keras

DEFAULT_EPOCH = 20
class KerasMLP:

  def __init__(self, args, data):

    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH
    self.batch  = args.input

    self.data = data

    self.model = keras.Sequential()
    
    """
      adds an input layer containing the number of attributes specified along with the -i flag
      and having the hyperbolic tangent as the activation function
    """
    self.model.add(keras.layers.Dense(args.input, activation="tanh"))

    """
      adds hidden layers containing units according to the value supplied with the -n flag
      e.g. -n 4 5 yields a list [4,5] therefore two hidden layers will be added. The former
      having 4 neurons and the latter 5 neurons
    """
    for neurons in args.neurons:
      self.model.add(keras.layers.Dense(neurons, activation="tanh"))

    """
      adds output input layer containing the number of attributes specified along with the -o flag
      and having the linear function as the activation function
    """
    self.model.add(keras.layers.Dense(args.output, activation="linear"))

    self.model.compile(
      optimizer=tf.train.GradientDescentOptimizer(args.learning),
      loss=keras.losses.MSE,
      metrics=[keras.metrics.MAE]
    )

  def run(self):

    self.model.fit(
      self.data['x_train_rows'], self.data['y_train_labels'], epochs=self.epochs, batch_size=self.batch,
      validation_data=(self.data['x_validate_rows'], self.data['y_validate_labels'])
    )

    self.model.predict(self.data['x_train_rows'], batch_size=self.batch)

    #print model.to_json()




