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

    self.learning_constant = args.learning

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

  
  def prepare_model(self):

    y = 0
    for index, number_neurons in enumerate(self.neurons_by_hidden_layer):
      if index == 0:
        input_ = self.X
      else:
        input_ = y
      
      label   = 'hidden'+str(index)
      weights = self.weights[label]
      bias    = self.biases[label]

      y = self.activation_function(tf.add(tf.matmul(input_, weights), bias))

    return self.output_function(y)

  def run(self):

    self.build()

    neural_network = self.prepare_model()

    """ the loss function is a MSE function
    and it evaluates the model (neural_network in this case)
    """
    loss_function = tf.reduce_mean(tf.square(self.Y - neural_network))

    """ alternatively you can use cross-entropy loss function, which is defined by
      tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=self.Y)
    """

    # defines the optimiser
    optimiser = tf.train.GradientDescentOptimizer(self.learning_constant)

    # training operation
    training_operation = optimiser.minimize(loss_function)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
      session.run(init)

      for epoch in range(self.epochs):
        print epoch
      
      session.close()
  
  """ Uses the hyperbolic tangent as activation function """
  def activation_function(self, x):
    return tf.tanh(x)
  
  """ Uses a linear function as output function """
  def output_function(self, y):
    return tf.add(tf.matmul(y, self.weights['output']), self.biases['output'])