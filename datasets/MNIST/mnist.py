import struct
import matplotlib.pyplot as plt


def load(path: str = "datasets/MNIST/"):
    with open(f"{path}train-images-idx3-ubyte", "rb") as f:
        # Read header (4 big-endian integers)
        magic = struct.unpack(">I", f.read(4))[0]
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        # Read all pixel data
        x_train = []

        for _ in range(num_images):
            image = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    pixel = struct.unpack("B", f.read(1))[0]
                    row.append(pixel)
                image.append(row)
            x_train.append(image)

    with open(f"{path}train-labels-idx1-ubyte", "rb") as f:
        # Read header
        magic = struct.unpack(">I", f.read(4))[0]
        num_labels = struct.unpack(">I", f.read(4))[0]

        y_train = []
        for _ in range(num_labels):
            label = struct.unpack("B", f.read(1))[0]
            y_train.append(label)

    with open(f"{path}t10k-images-idx3-ubyte", "rb") as f:
        # Read header (4 big-endian integers)
        magic = struct.unpack(">I", f.read(4))[0]
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        # Read all pixel data
        x_test = []

        for _ in range(num_images):
            image = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    pixel = struct.unpack("B", f.read(1))[0]
                    row.append(pixel)
                image.append(row)
            x_test.append(image)

    with open(f"{path}t10k-labels-idx1-ubyte", "rb") as f:
        # Read header
        magic = struct.unpack(">I", f.read(4))[0]
        num_labels = struct.unpack(">I", f.read(4))[0]

        y_test = []
        for _ in range(num_labels):
            label = struct.unpack("B", f.read(1))[0]
            y_test.append(label)

    return x_train, y_train, x_test, y_test

def show_image(image):
    """
    image: 2D list (28x28) with values 0-255
    """

    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

# x_train, y_train, x_test, y_test = load()

# print(len(x_train))         # 60000
# print(len(x_train[0]))      # 28
# print(len(x_train[0][0]))   # 28
# print(len(y_train))         # 60000

# print()

# print(len(x_test))         # 10000
# print(len(x_test[0]))      # 28
# print(len(x_test[0][0]))   # 28
# print(len(y_test))         # 10000
