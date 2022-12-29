import math
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1 / (1 + math.e ** (-x))

    return y

max_value = 3
x = range(-max_value, max_value+1)
y = []

for i in range(-max_value, max_value+1):
    y.append(sigmoid(i))

fig = plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.show()
