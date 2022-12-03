import matplotlib.pyplot as plt
import numpy
import numpy as np

data = np.load('experiments/distribution.npy', allow_pickle=True)

iter0 = data[0]
iter19 = data[19]
iter39 = data[39]
iter59 = data[59]
iter79 = data[79]
iter99 = data[99]

length = len(data[0])

fig = plt.figure(figsize=(12, 6))

x = range(length)

plt.plot(x, iter0, c='blue', label='iter0')
plt.plot(x, iter19, c='red', label='iter19')
plt.plot(x, iter39, c='yellow', label='iter39')
plt.plot(x, iter59, c='green', label='iter59')
plt.plot(x, iter79, c='cyan', label='iter79')
plt.plot(x, iter99, c='pink', label='iter99')

plt.show()
