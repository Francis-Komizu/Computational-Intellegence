import numpy as np
import utils
import models
import random
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

seed = 4044
random.seed(seed)
np.random.seed(seed)

# 超参数
population_size = 100
crossover_prob = 0.8
mutation_prob = 0.2
memory_size = 5
crossover_mode = 'OX'

iterations = 4000

file_name = 'ch130'
exp_name = 'ga-sigmoid-mem'
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
                                   crossover_prob=crossover_prob,
                                   mutation_prob=mutation_prob,
                                   memory_size=memory_size,
                                   crossover_mode=crossover_mode)

    Population.init_population()
    best_seq = Population.best_chromosome.gene_seq

    distances = []

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

        # print(Population.generation)

    # utils.save_result(distances, save_path)
    total_distances.append(distances)

total_distances = np.array(total_distances)
mean_distances = total_distances.mean(axis=0)
utils.save_result(mean_distances, save_path)
