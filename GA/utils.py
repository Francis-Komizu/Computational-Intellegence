import numpy as np
import matplotlib.pyplot as plt
import math


def get_cities(file_path: str):
    cities = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        city_num = len(lines)
        for line in lines:
            line = line.strip('\n')
            parts = line.split(' ')
            data = []
            for part in parts:
                if part:
                    data.append(part)

            city, x, y = data[0], data[1], data[2]
            x, y = float(x), float(y)
            cities[city] = np.array([x, y], dtype=np.int16)

    return cities, city_num


def get_distance_matrix(cities: dict, city_num: int):
    distance_matrix = np.zeros([city_num, city_num], dtype=np.int16)
    for i in range(1, city_num + 1):
        for j in range(1, city_num + 1):
            distance = np.linalg.norm(cities[str(i)] - cities[str(j)],
                                      axis=0,
                                      ord=2)
            distance_matrix[i - 1][j - 1] = distance

    return distance_matrix


def compute_optimal_distance(file_path: str, distance_matrix):
    solution = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            city = line.strip('\n')
            solution.append(city)

        length = len(solution)
        solution = np.array(solution, dtype=np.int16)

    optimal_distance = compute_distance(solution, length, distance_matrix)

    return optimal_distance


def compute_distance(solution, length, distance_matrix):
    total_distance = 0
    for i in range(length):
        city_cur = solution[i]
        if i < length - 1:
            city_next = solution[i + 1]
        else:
            # 回到起始点
            city_next = solution[0]

        distance = distance_matrix[city_cur - 1][city_next - 1]
        total_distance += distance

    return total_distance


def sigmoid(x):
    y = 1 / (1 + math.e ** (-x))

    return y


def visualize_result(distances):
    iterations = range(len(distances))

    fig = plt.figure(figsize=(12, 6))
    plt.title('Distance Changing Curve')
    plt.plot(iterations, distances)
    plt.xlabel('iterations')
    plt.ylabel('distance')
    plt.show()


def save_result(distances: list, save_path: str):
    result = np.array(distances)
    np.save(save_path, result)


def load_result(result_path: str):
    result = np.load(result_path)
    return result


if __name__ == '__main__':
    pass
