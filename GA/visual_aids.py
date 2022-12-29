import matplotlib.pyplot as plt
import numpy as np
from utils import *

cities, city_num = get_cities('data/ch130.tsp')
distance_matrix = get_distance_matrix(cities, city_num)
optimal_distance = compute_optimal_distance('data/ch130.opt.tour', distance_matrix)

# 加载实验数据
ga_vanilla = np.load('experiments/ga-vanilla-mem-ch130.npy', allow_pickle=True)
ga_sigmoid = np.load('experiments/ga-sigmoid-mem-ch130.npy', allow_pickle=True)


length = len(ga_vanilla)

optimal = np.zeros(length, dtype=np.int32)
for i in range(length):
    optimal[i] = optimal_distance

iterations = range(length)

fig = plt.figure(figsize=(12, 6))
plt.title('Distance Changing Curve')
plt.plot(iterations, ga_vanilla, c='blue', label='GA vanilla')
plt.plot(iterations, ga_sigmoid, c='skyblue', label='GA + sigmoid')
# plt.plot(iterations, ga_vanilla_mem, c='green', label='GA vanilla + mem')
# plt.plot(iterations, ga_sigmoid_mem, c='orangered', label='GA + sigmoid + mem')
# plt.plot(iterations, ga_scx, c='sandybrown', label='GA + SCX')
# plt.plot(iterations, ga_mscx, c='springgreen', label='GA + MSCX')
# plt.plot(iterations, ga_mscx_radius, c='purple',label='GA + MSCX_Radius')
# plt.plot(iterations, iga_vanilla, c='red', label='IGA vanilla')
# plt.plot(iterations, iga_mscx, c='fuchsia', label='IGA + MSCX')
plt.plot(iterations, optimal, '--', c='darkviolet', label='optimal')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.legend()
plt.show()