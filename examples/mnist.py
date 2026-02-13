import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import flatten
from datasets.MNIST import mnist
from mlp import MLP


# Gathering data
x_train_raw, y_train_raw, x_test_raw, y_test_raw = mnist.load()

# Pre-processing
x_train = flatten(x_train_raw)[:100]
y_train = [[y] for y in y_train_raw][:100]
x_test = flatten(x_test_raw)[:10]
y_test = [[y] for y in y_test_raw][:10]

# Instantiating model
mlp = MLP(
    hidden_layer_sizes=[16, 16],
    activation_functions=["sigmoid", "sigmoid", "sigmoid"],
    cost_function="mse",
    learning_rate=0.1,
    num_epochs=1000
)

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
