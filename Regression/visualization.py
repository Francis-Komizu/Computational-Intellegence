import matplotlib.pyplot as plt
import numpy
import numpy as np
import csv
import torch


def plot_histogram(path, column=0, bins=10, log=False, normalize=''):
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, column].astype(float)

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


def plot_training_curve(train_loss_path, dev_loss_path):
    train_loss = np.load(train_loss_path, allow_pickle=True)
    dev_loss = np.load(dev_loss_path, allow_pickle=True)

    train_length = len(train_loss)
    dev_length = len(dev_loss)

    scale = train_length / dev_length

    train_steps = range(train_length)
    dev_steps = []
    for i in range(dev_length):
        dev_steps.append((i + 1) * scale)

    fig = plt.figure(figsize=(6, 4))
    plt.title('Training Loss Curve')
    plt.plot(train_steps, train_loss, c='red', label='training loss')
    plt.plot(dev_steps, dev_loss, c='cyan', label='validation loss')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_pred_and_gt(pred_path, gt_path):
    with open(pred_path, 'r') as fp:
        preds = list(csv.reader(fp))
        preds = np.array(preds[1:]).astype(float)

    with open(gt_path, 'r') as fp:
        gt = list(csv.reader(fp))
        gt = np.array(gt[1:])[:, 3].astype(float)

    fig = plt.figure(figsize=(6, 4))

    max_value = max(max(preds), max(gt))
    steps = range(int(max_value))

    plt.plot(gt, preds, 'o', c='red')
    plt.plot(steps, steps, c='blue')
    plt.xlabel('ground truth')
    plt.ylabel('predicted')
    plt.show()


if __name__ == '__main__':
    train_loss_path = 'experiments/train_loss.npy'
    dev_loss_path = 'experiments/dev_loss.npy'
    pred_path = 'experiments/preds.csv'
    gt_path = 'data/test_set.csv'
    train_path = 'data/train_set.csv'

    plot_pred_and_gt(pred_path, gt_path)

    plot_training_curve(train_loss_path, dev_loss_path)

    plot_histogram(train_path, 0, 20, log=False, normalize='zero')
