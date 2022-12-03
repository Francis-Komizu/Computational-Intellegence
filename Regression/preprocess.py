import csv
import numpy as np
import random


def csv_reader(path):
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:]).astype(float)

    return data


def csv_writer(data, path):
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['RM', 'LSTAT', 'PTRATIO', 'MEDV'])
        for i, row in enumerate(data):
            writer.writerow([row[0], row[1], row[2], row[3]])


def data_spliter(raw_data):
    train_set = []
    dev_set = []
    test_set = []
    for data in raw_data:
        rand = random.random()
        if rand < 0.8:  # 训练集
            train_set.append(data)
        elif 0.8 <= rand < 0.9:  # 验证集
            dev_set.append(data)
        else:  # 测试集
            test_set.append(data)

    return train_set, dev_set, test_set


if __name__ == '__main__':
    raw_path = 'data/housing.csv'
    train_path = 'data/train_set.csv'
    dev_path = 'data/dev_set.csv'
    test_path = 'data/test_set.csv'

    raw_data = csv_reader(raw_path)
    train_set, dev_set, test_set = data_spliter(raw_data)

    csv_writer(train_set, train_path)
    csv_writer(dev_set, dev_path)
    csv_writer(test_set, test_path)
