import random

class MLP():
    # Methods
    def __init__(self, hidden_layer_sizes: list[int] = [2, 4, 3], activation_functions: list[str] = ["sigmoid", "sigmoid", "sigmoid", "tanh"], cost_function: str = "MSE", learning_rate: float = 0.1, num_epochs: int = 10000):
        # Hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_functions = [af_map[activation_function.lower()] for activation_function in activation_functions]
        self.activation_functions_derivatives = [afd_map[activation_function.lower()] for activation_function in activation_functions]
        self.cost_function = cf_map[cost_function.lower()]
        self.cost_function_derivative = mse_derivative
        self.learning_rate = learning_rate
        self.n_epochs = num_epochs # number of epochs

        # Parameters
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        # Internal attributes
        self.input_size = 0
        self.output_size = 0

        # Shortenings
        self.activ = self.activation_functions
        self.activ_der = self.activation_functions_derivatives
        self.cost = self.cost_function
        self.cost_der = self.cost_function_derivative
        self.lr = self.learning_rate

    # --- Main methods ---

    def fit(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.randomize_params()

    def train(self, x_train: list[list[float]], y_train: list[list[float]]):
        for epoch in range(self.n_epochs):
            self.backwardprop(x_train, y_train)

    def test(self, x_test: list[list[float]], y_test: list[list[float]]) -> tuple[list[list[float]], list[float]]:
        """ Returns predictions and accuracies. """

        y_pred = self.predict(x_test)

        accuracies: list[float] = []
        for sample in range(len(y_test)):
            accuracies.append(1 - self.cost(y_pred[sample], y_test[sample]))

        return y_pred, accuracies

    def predict(self, x_pred: list[list[float]]) -> list[list[float]]:
        outputs, zs = self.forwardprop(x_pred)
        return [outputs[sample][-1] for sample in range(len(outputs))]

    # --- Useful methods ---

    def n_layers(self) -> int:
        return len(self.hidden_layer_sizes) + 1

    def layer(self, key: int) -> int:
        """ Returns the number of neurons in that layer. If key == -1, returns input size. """

        if key == len(self.hidden_layer_sizes):
            return self.output_size
        if key == -1:
            return self.input_size
        return self.hidden_layer_sizes[key]

    # --- Internal methods ---

    def forwardprop(self, x_pred: list[list[float]]) -> list[list[list[float]]]:
        outputs: list[list[list[float]]] = []
        zs: list[list[list[float]]] = []
        for sample in range(len(x_pred)):
            outputs.append([])
            zs.append([])
            for layer in range(self.n_layers()):
                outputs[sample].append([])
                zs[sample].append([])
                for neuron in range(self.layer(layer)):
                    z = 0
                    for previous_neuron in range(self.layer(layer - 1)):
                        if layer == 0: # first hidden layer
                            previous_output = x_pred[sample][previous_neuron]
                        else:
                            previous_output = outputs[sample][layer - 1][previous_neuron]
                        z += self.weights[layer][neuron][previous_neuron] * previous_output
                    z += self.biases[layer][neuron]
                    zs[sample][layer].append(z)
                    outputs[sample][layer].append(self.activ[layer](z))
        return outputs, zs

    def backwardprop(self, x_train: list[list[float]], y_train: list[list[float]]):
        outputs, zs = self.forwardprop(x_train)

        # Calculating deltas
        deltas: list[list[list[float]]] = []
        for sample in range(len(x_train)):
            deltas.append([])
            for layer in range(self.n_layers() - 1, -1, -1): # starting at the last layer and going back.
                i = self.n_layers() - 1 - layer # counter
                deltas[sample].append([])
                for neuron in range(self.layer(layer)):
                    neuron_output = outputs[sample][layer][neuron]
                    if layer == self.n_layers() - 1: # output layer
                        layer_error = self.cost_der(neuron_output, y_train[sample][neuron])
                    else:
                        layer_error = 0
                        for next_neuron in range(self.layer(layer + 1)):
                            layer_error += deltas[sample][i - 1][next_neuron] * self.weights[layer + 1][next_neuron][neuron]
                    deltas[sample][i].append(layer_error * self.activ_der[layer](neuron_output))
            deltas[sample].reverse()

        # Updating parameters
        for sample in range(len(x_train)):
            for layer in range(self.n_layers()):
                for neuron in range(self.layer(layer)):
                    for previous_neuron in range(self.layer(layer - 1)):
                        previous_output = outputs[sample][layer - 1][previous_neuron] if layer != 0 else x_train[sample][previous_neuron]
                        self.weights[layer][neuron][previous_neuron] += -(deltas[sample][layer][neuron] * previous_output) * self.lr
                    self.biases[layer][neuron] += -(deltas[sample][layer][neuron]) * self.lr

    def randomize_params(self):
        self.weights = []
        self.biases = []
        for layer in range(self.n_layers()):
            self.weights.append([])
            self.biases.append([])
            for neuron in range(self.layer(layer)):
                self.biases[layer].append(random.random())
                self.weights[layer].append([])
                for previous_neuron in range(self.layer(layer - 1)):
                    self.weights[layer][neuron].append(random.random())

# Constants
e = 2.718281828459045235360287471352

# Activation functions
def relu(x: float) -> float:
    return max(0, x)

def sigmoid(x: float) -> float:
    return 1 / (1 + (e ** -x))

def tanh(x: float) -> float:
    return ((e ** (2 * x)) - 1) / ((e ** (2 * x)) + 1)

af_map = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh}

# Activation function derivatives
def relu_derivative(x: float) -> float:
    return 0 if x <= 0 else 1

def sigmoid_derivative(x: float) -> float:
    sig = sigmoid(x)
    return sig * (1 - sig)

def sigmoid_derivative_f(x: float) -> float:
    """ This is not the derivative of the sigmoid function.
    This function returns sigmoid_derivative(x) when you pass sigmoid(x) to it.
    Useful when you already have the sigmoid and doesn't want to recompute it. """

    return x * (1 - x)

def tanh_derivative(x: float) -> float:
    return 4 / (((e ** -x) + (e ** x)) ** 2)

afd_map = {"relu": relu_derivative, "sigmoid": sigmoid_derivative_f, "tanh": tanh_derivative}

# Cost functions
def mse(y_pred: list[float], y_test: list[float]) -> float:
    error = 0
    for i in range(len(y_pred)):
        error += (y_pred[i] - y_test[i]) ** 2
    error /= len(y_pred)
    return error

def rmse(y_pred: list[float], y_test: list[float]) -> float:
    import math
    return math.sqrt(mse(y_pred, y_test))

def mae(y_pred: list[float], y_test: list[float]) -> float:
    error = 0
    for i in len(y_pred):
        error += abs(y_pred[i] - y_test[i])
    error /= len(y_pred)
    return error

# Cost function derivatives
def mse_derivative(y_pred: float, y_test: float) -> float:
    return 2 * (y_pred - y_test)

cf_map = {"mse": mse, "rmse": rmse, "mae": mae}
