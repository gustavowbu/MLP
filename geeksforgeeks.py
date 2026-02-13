import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) # 2x4
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) # 4x1

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden # 4x4 = 4x2 . 2x4 + 4
        self.hidden_output = self.sigmoid(self.hidden_activation) # 4x4 = 4x4

        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output # 4x1 = 4x4 . 4x1 + 4
        self.predicted_output = self.sigmoid(self.output_activation) # 4x1 = 4x1

        return self.predicted_output # 4x1

    def backward(self, X, y, learning_rate):
        output_error = 2 * (self.predicted_output - y) # 4x1 = 4x1 - 4x1
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output) # 4x1 = 4x1 * 4x1

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T) # 4x4 = 4x1 . 1x4
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output) # 4x4 = 4x4 * 4x4

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * -learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * -learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * -learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * -learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4x2
y = np.array([[0], [1], [1], [0]]) # 4x1

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# nn.weights_input_hidden = np.array([[0.8, 0.7], [0.5, 0.4], [0.7, 0], [0.8, 0.4]]).T
# nn.weights_hidden_output = np.array([[0, 0.4, 0.7, 0.2]]).T
# nn.bias_hidden = np.array([[0.1, 0.2, 0.6, 0]])
# nn.bias_output = np.array([[0.7]])

# print("Weights and biases:")
# print(nn.weights_input_hidden)
# print(nn.weights_hidden_output)
# print(nn.bias_hidden)
# print(nn.bias_output)
# print()

nn.train(X, y, epochs=10000, learning_rate=0.1)

output = nn.feedforward(X)
print("Predictions after training:")
print(output)
