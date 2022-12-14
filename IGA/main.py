import numpy as np
import utils
import models
import random
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm

seed = 4044
random.seed(seed)
np.random.seed(seed)

# 超参数
population_size = 100
alpha = 1.0
beta = 0.0
delta = 0.0
crossover_prob = 0.8
pr = 0.5
mutation_prob = 0.2
memory_size = 5
radius = 1
crossover_mode = 'OX'

iterations = 4000

file_name = 'ch130'
exp_name = 'ga-vanilla-mem'
exp_num = 5
data_path = f'data/{file_name}.tsp'
save_path = f'experiments/{exp_name}-{file_name}.npy'

# initialization
cities, city_num = utils.get_cities(data_path)
distance_matrix = utils.get_distance_matrix(cities, city_num)

total_distances = []

for i in range(exp_num):
    Population = models.Population(chro_num=population_size,
                                   gene_num=city_num,
                                   distance_matrix=distance_matrix,
                                   alpha=alpha,
                                   beta=beta,
                                   delta=delta,
                                   crossover_prob=crossover_prob,
                                   mutation_prob=mutation_prob,
                                   memory_size=memory_size,
                                   radius=radius,
                                   crossover_mode=crossover_mode)

    Population.init_population()
    distances = []  # 存储每代的最优结果，用于可视化

    # result = []

    # 迭代
    for i in tqdm(range(iterations)):

        # 选择
        Population.selection()

        # 交叉
        Population.crossover()

        # 变异
        Population.mutation()

        # 记忆
        Population.memorization()

        # 更新
        Population.update()

        # 记录每代的最短距离
        best_seq = Population.best_chromosome.gene_seq
        best_distance = utils.compute_distance(best_seq, city_num, distance_matrix)
        distances.append(best_distance)

        length = len(Population.chromosomes)

        # affinities = []
        #
        # for chromosome in Population.chromosomes:
        #     affinities.append(chromosome.affinity)
        #
        # affinities = np.array(affinities)
        # max, min = np.max(affinities), np.min(affinities)
        #
        # affinities = (affinities - min) / (max - min)
        #
        # result.append(affinities)

        # print(Population.generation)

    total_distances.append(distances)

    # save_path = 'experiments/distribution.npy'
    # utils.save_result(distances, save_path)
    # utils.visualize_result(distances)

total_distances = np.array(total_distances)
mean_distances = total_distances.mean(axis=0)
utils.save_result(mean_distances, save_path)
