def flatten(data: list[list[list[float]]]) -> list[list[float]]:
    flattened: list[list[float]] = []
    for sample in range(len(data)):
        flattened.append([])
        for row in range(len(data[sample])):
            flattened[sample].extend(data[sample][row])

    return flattened

def flatten_sample(sample: list[list[float]]) -> list[float]:
    flattened: list[float] = []
    for row in range(len(sample)):
        flattened.extend(sample[row])

    return flattened

# x = [
#     [[1, 2], [3, 4]],
#     [[5, 6], [7, 8]],
#     [[9, 10], [11, 12]]
# ]

# print(x)
# print(flatten(x))
# print([flatten_sample(sample) for sample in x])
