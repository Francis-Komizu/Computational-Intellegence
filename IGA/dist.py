import matplotlib.pyplot as plt
import numpy
import numpy as np

data = np.load('experiments/distribution.npy', allow_pickle=True)


def plot_histogram(data, bins=50, log=False, normalize=''):
    if log:
        data = numpy.log(data)

    if normalize == 'zero':
        mean, std = numpy.mean(data, axis=0), numpy.std(data, axis=0)
        data = (data - mean) / std
    elif normalize == 'linear':
        max, min = numpy.max(data, axis=0), numpy.min(data, axis=0)
        data = (data - min) / (max - min)
    else:
        pass

    fig = plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.show()


plot_histogram(data[999], normalize='linear')
