import numpy as np
from utils import *
file_name = 'ch130'

ga_vanilla = np.load(f'experiments/ga-vanilla-{file_name}.npy', allow_pickle=True)
ga_sigmoid = np.load(f'experiments/ga-sigmoid-{file_name}.npy', allow_pickle=True)

cities, city_num = get_cities(f'data/{file_name}.tsp')
distance_matrix = get_distance_matrix(cities, city_num)
optimal_distance = compute_optimal_distance(f'data/{file_name}.opt.tour', distance_matrix)

print(ga_vanilla[-1])
print(ga_sigmoid[-1])
print(optimal_distance)