# MultiLayer Perceptron

A MultiLayer Perceptron built with Tensorflow and Python.

### Running

First clone or download this repository, then run:

```
$ python main.py -i <n_inputs> -hl <n_hidden_layers> -o <n_outputs> -n <neurons>
```
The -n argument takes a list of numbers separated by space, e.g. "-n 4 5". This means 4 neurons will be assigned to the first hidden layer and 5 to the second hidden layers. 

If the number of items in the neuron's list is greater than the supplied number of hidden layers a ValueError is raised.

For more information on the parameters see
```
$ python main.py --help
```