import tensorflow as tf
from tensorflow import keras

DEFAULT_EPOCH = 20

class MultiLayerPerceptron:

  def __init__(self, args, data):
    self.number_input = args.input
    self.number_hidden_layers = args.hidden
    self.number_output = args.output
    self.neurons_by_hidden_layer = args.neurons
    self.epochs = args.epochs if args.epochs else DEFAULT_EPOCH

    self.data = data

    self.X = tf.placeholder(tf.float32, [None, self.number_input])
    self.Y = tf.placeholder(tf.float32, [None, self.number_output])

    self.weights = {}
    self.biases = {}


  def build(self):
    
    print 'Number of inputs: ' + str(self.number_input)
    print 'Number of hidden layers: ' + str(self.number_hidden_layers)
    print 'Number of outputs: ' + str(self.number_output)
    print 'List of neurons per hidden layer: ' + str(self.neurons_by_hidden_layer)

    for index, number_neurons in enumerate(self.neurons_by_hidden_layer):
      
      label = 'hidden' + str(index)

      if index == 0:
        current_layer = self.number_input
        next_layer = number_neurons
      else:
        current_layer = self.neurons_by_hidden_layer[index-1]
        next_layer = number_neurons
      
      self.weights[label] = tf.Variable(
        tf.random_normal([current_layer, next_layer]),
        name=label+"_weights")
    
      self.biases[label] = tf.Variable(
        tf.random_normal([number_neurons]),
        name=label+"_biases")

    self.weights['output'] = tf.Variable(
      tf.random_normal([self.neurons_by_hidden_layer[self.number_hidden_layers-1], self.number_output]),
      name="output_weights")
    
    self.biases['output'] = tf.Variable(
      tf.random_normal([self.number_output]),
      name="output_biases")

    

  def run(self):

    init = tf.global_variables_initializer()

    with tf.Session() as session:
      session.run(init)

      session.close()
  
  def activation_function(self, x):
    return tf.tanh(x, name=None)
  
  def output_function(self, y):
    return tf.add(tf.matmul(y, self.weights), self.biases)