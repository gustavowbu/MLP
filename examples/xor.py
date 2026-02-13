import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mlp import MLP


# Gathering data
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Instantiating model
mlp = MLP(
    hidden_layer_sizes=[4],
    activation_functions=["sigmoid", "sigmoid"],
    cost_function="mse",
    learning_rate=0.1,
    num_epochs=10000
)
# mlp.weights = [[[0.8, 0.7], [0.5, 0.4], [0.7, 0], [0.8, 0.4]], [[0, 0.4, 0.7, 0.2]]]
# mlp.biases = [[0.1, 0.2, 0.6, 0], [0.7]]

# Training model
mlp.fit(input_size=len(x_train[0]), output_size=len(y_train[0]))
mlp.train(x_train, y_train)

# Testing model
y_pred, accuracy = mlp.test(x_train, y_train)

# Showing results
print(f"Predictions:")
for sample in range(len(x_train)):
    print(f"  {x_train[sample][0]} xor {x_train[sample][1]} = {y_pred[sample][0]:.2f}")
    print(f"    Correct value: {y_train[sample][0]} | Accuracy: {accuracy[sample] * 100:.2f}%")
