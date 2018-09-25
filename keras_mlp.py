import tensorflow as tf
from tensorflow import keras

DEFAULT_EPOCH = 20
class KerasMLP:

  def __init__(self, args, data):

    self.data = data

    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH
    self.batch  = args.input

    self.model = keras.Sequential()
    
    """
        Adds an input layer containing the number of attributes specified along with the -i flag
      and having the hyperbolic tangent as the activation function.
        Considering a neuron list with the format [4,5], its output shape ought to be (*, 4).
    """
    self.model.add(keras.layers.Dense(args.neurons[0], activation="tanh", input_shape=(args.input,)))

    """
        Adds hidden layers containing units according to the value supplied with the -n flag
      e.g. -n 4 5 yields a list [4,5] therefore two hidden layers will be added. The former
      having 4 neurons and the latter 5 neurons.
        Their output shapes ought to be respectively (*, 5) and (*, args.output).
    """
    for key, neurons in enumerate(args.neurons):
      if key < len(args.neurons) - 1:
        self.model.add(keras.layers.Dense(args.neurons[key+1], activation="tanh"))
      else:
        self.model.add(keras.layers.Dense(args.output, activation="tanh"))

    """
        Adds output input layer containing the number of attributes specified along with the -o flag
      and having the linear function as the activation function.
        Its output shape ought to be (*, args.output).
    """
    self.model.add(keras.layers.Dense(args.output, activation="linear"))

    self.model.compile(
      optimizer=tf.train.GradientDescentOptimizer(args.learning),
      loss=keras.losses.MSE,
      metrics=[keras.metrics.MAE]
    )

  def run(self):
    
    train_data = self.data['x_train_rows']
    train_label = self.data['y_train_labels']

    self.model.fit(
      train_data, train_label, epochs=self.epochs, batch_size=self.batch,
      validation_data=(self.data['x_validate_rows'], self.data['y_validate_labels'])
    )

    self.model.predict(self.data['x_train_rows'], batch_size=self.batch)

    #print model.to_json()




