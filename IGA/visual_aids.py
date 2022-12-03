import matplotlib.pyplot as plt
import numpy as np

# 加载实验数据
ga_vanilla = np.load('experiments/GA-1-OX-200-6000.npy', allow_pickle=True)
ga_mem = np.load('experiments/GA-2-OX-200-6000.npy', allow_pickle=True)
ga_sim = np.load('experiments/GA-3-OX-200-6000.npy', allow_pickle=True)
ga_mox = np.load ('experiments/GA-4-MOX-200-6000.npy', allow_pickle=True)
ga_scx = np.load('experiments/GA-5-SCX-200-6000.npy')
ga_mscx = np.load('experiments/GA-6-MSCX-200-6000.npy', allow_pickle=True)
ga_mscx_radius = np.load('experiments/GA-7-MSCX_Radius-200-6000.npy', allow_pickle=True)
iga_vanilla = np.load('experiments/IGA-1-OX-200-6000.npy', allow_pickle=True)
iga_mscx = np.load('experiments/IGA-2-MSCX-200-6000.npy')

length = len(ga_vanilla)
optimal = np.zeros(length, dtype=np.int32)
for i in range(length):
    optimal[i] = 33503

iterations = range(length)

fig = plt.figure(figsize=(12, 6))
plt.title('Distance Changing Curve')
plt.plot(iterations, ga_vanilla, c='blue', label='GA vanilla')
plt.plot(iterations, ga_mem, 'o', c='skyblue', label='GA + MEM')
plt.plot(iterations, ga_sim, c='green', label='GA + SIM')
plt.plot(iterations, ga_mox, c='orangered', label='GA + MOX')
plt.plot(iterations, ga_scx, c='sandybrown', label='GA + SCX')
plt.plot(iterations, ga_mscx, c='springgreen', label='GA + MSCX')
plt.plot(iterations, ga_mscx_radius, c='purple',label='GA + MSCX_Radius')
plt.plot(iterations, iga_vanilla, c='red', label='IGA vanilla')
plt.plot(iterations, iga_mscx, c='fuchsia', label='IGA + MSCX')
plt.plot(iterations, optimal, '--', c='darkviolet', label='optimal')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.legend()
plt.show()