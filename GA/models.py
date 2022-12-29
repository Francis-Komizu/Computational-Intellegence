import numpy as np
import random
import utils


# 染色体/抗体
class Chromosome:
    def __init__(self, gene_num=0, distance_matrix=None):
        self.gene_num = gene_num  # 基因数目
        self.distance_matrix = distance_matrix
        self.gene_seq = np.arange(1, gene_num + 1)  # 基因序列
        self.distance = float('inf')

    def init_chromosome(self):
        """
        随机初始化基因序列
        1. 产生顺序序列
        2. 循环并产生介于0至gene_num-1之间的随机数
        3. 将当前gene与remainder对应的gene交换
        """
        for i in range(self.gene_num):
            j = random.randint(0, self.gene_num - 1)
            temp = self.gene_seq[j]
            self.gene_seq[j] = self.gene_seq[i]
            self.gene_seq[i] = temp

    def display_chromosome(self):
        info = f'{self.gene_seq}'
        print(info)

    def compute_distance(self):
        distance = utils.compute_distance(self.gene_seq,
                                          self.gene_num,
                                          self.distance_matrix)
        self.distance = distance

    def get_distance(self):
        return self.distance

    def duplication(self, chro):
        self.gene_num = chro.gene_num
        self.distance_matrix = chro.distance_matrix
        self.gene_seq = chro.gene_seq
        self.distance = chro.distance

    def crossover_OX(self, chro):
        """
        Older Crossover
        1. 随机产生两个交叉点 (cross point)
        2. P1介于交叉点之间的基因片段直接复制到子代相同位置
        3. P2从第二个交叉点开始顺序遍历，将没有出现在子代的基因加入到子代中
        4. 认为self对应P1而chro对应P2
        """
        child = Chromosome(self.gene_num, self.distance_matrix)
        child_seq = np.zeros(self.gene_num, dtype=np.int32)  # 子代个体
        xp1 = random.randint(0, self.gene_num - 1)  # cross point 1
        xp2 = random.randint(0, self.gene_num - 1)  # cross point 2
        if xp1 > xp2:
            temp = xp1
            xp1 = xp2
            xp2 = temp

        # P1的片段直接复制到子代
        for i in range(xp1, xp2 + 1):
            child_seq[i] = self.gene_seq[i]

        # P2中不在child里出现的基因顺序复制到child
        i = j = (xp2 + 1) % self.gene_num  # i, j分别是P2和child的伪指针
        count = 0  # 记录child是否已经填满
        while True:
            if chro.gene_seq[i] not in child_seq:
                child_seq[j] = chro.gene_seq[i]
                i = (i + 1) % self.gene_num
                j = (j + 1) % self.gene_num
                count += 1
            else:
                i = (i + 1) % self.gene_num

            # child填满退出
            if count == self.gene_num - (xp2 - xp1 + 1):
                break

        child.gene_seq = child_seq
        child.compute_distance()

        return child, xp1, xp2

    def crossover_MOX(self, chro):
        """
        Modified Older Crossover
        在older crossover的基础上，将子代交叉点开外的两个基因片段反向 (flip)
        """
        child, xp1, xp2 = self.crossover_OX(chro)

        if xp1 != 0:
            mid = (xp1 - 1) // 2
            for i in range(0, mid):
                self.swap_gene(child, i, xp1 - 1 - i)

        if xp2 != self.gene_num:
            mid = (xp2 + 1 + self.gene_num) // 2
            for i in range(xp2 + 1, mid):
                self.swap_gene(child, i, xp2 + 1 + self.gene_num - 1 - i)

        child.compute_distance()

        return child

    def crossover_SCX(self, chro):
        """
        Sequential Constructive Crossover
        1. 基因p = Parent1(1)，并将其作为子代第一个基因
        2. 按顺序搜索Parent1和Parent2中接在p后面的基因，即合法基因，并标记为visited
        如果没有合法基因，则从{1, 2, 3, ..., N}中按顺序找出第一个合法基因
        3. 从Parent1和Parent2中找到了a和b两个基因，然后计算Cost(p, a)和Cost(p, b)
        如果Cost(p, a) < Cost(p, b)，则将b接到子代中并把b赋给p
        4. 循环，直至子代拼成了完整的染色体
        5. 此处要建立基因值到索引的映射
        """

        parent1 = {}  # key为基因，value为索引
        parent2 = {}
        for i in range(self.gene_num):
            gene1 = self.gene_seq[i]
            gene2 = chro.gene_seq[i]
            parent1[gene1] = i
            parent2[gene2] = i

        auxiliary_seq = range(1, self.gene_num + 1)
        idx_p = 0
        gene_p = self.gene_seq[idx_p]
        child = Chromosome(self.gene_num, self.distance_matrix)
        child_seq = np.zeros(self.gene_num, dtype=np.int32)
        child_seq[0] = self.gene_seq[0]
        count = 1  # 判断子代是否填满

        idx_a = (parent1[gene_p] + 1) % self.gene_num
        idx_b = (parent2[gene_p] + 1) % self.gene_num

        while count < self.gene_num:
            if self.gene_seq[idx_a] not in child_seq:
                gene_a = self.gene_seq[idx_a]
            else:
                for gene in auxiliary_seq:
                    if gene not in child_seq:
                        gene_a = gene

            if self.gene_seq[idx_b] not in child_seq:
                gene_b = self.gene_seq[idx_b]
            else:
                for gene in auxiliary_seq:
                    if gene not in child_seq:
                        gene_b = gene

            distance_a = self.distance_matrix[gene_p - 1][gene_a - 1]
            distance_b = self.distance_matrix[gene_p - 1][gene_b - 1]

            if distance_a < distance_b:
                child_seq[count] = gene_a
                gene_p = gene_a
                idx_a = parent1[gene_a]
            else:
                child_seq[count] = gene_b
                gene_p = gene_b
                idx_b = parent2[gene_b]

            count += 1

        child.gene_seq = child_seq
        child.compute_distance()

        return child

    def crossover_MSCX(self, chro):
        """
        Modified Sequential Constructive Crossover
        1. 基因p = Parent1(1)，并将其作为子代第一个基因
        2. 按顺序搜索Parent1和Parent2中接在p后面的基因，即合法基因，并标记为visited
        如果没有合法基因，则循环从该Parent中按顺序找出第一个合法基因
        3. 从Parent1和Parent2中找到了a和b两个基因，然后计算Cost(p, a)和Cost(p, b)
        如果Cost(p, a) < Cost(p, b)，则将b接到子代中并把b赋给p
        4. 循环，直至子代拼成了完整的染色体
        5. 此处要建立基因值到索引的映射
        """

        parent1 = {}  # key为基因，value为索引
        parent2 = {}
        for i in range(self.gene_num):
            gene1 = self.gene_seq[i]
            gene2 = chro.gene_seq[i]
            parent1[gene1] = i
            parent2[gene2] = i

        idx_p = 0
        gene_p = self.gene_seq[idx_p]
        child = Chromosome(self.gene_num, self.distance_matrix)
        child_seq = np.zeros(self.gene_num, dtype=np.int32)
        child_seq[0] = self.gene_seq[0]
        count = 1  # 判断子代是否填满

        idx_a = (parent1[gene_p] + 1) % self.gene_num
        idx_b = (parent2[gene_p] + 1) % self.gene_num

        while count < self.gene_num:
            while True:
                gene_a = self.gene_seq[idx_a]
                if gene_a not in child_seq:
                    break
                idx_a = (idx_a + 1) % self.gene_num

            while True:
                gene_b = self.gene_seq[idx_b]
                if gene_b not in child_seq:
                    break
                idx_b = (idx_b + 1) % self.gene_num

            distance_a = self.distance_matrix[gene_p - 1][gene_a - 1]
            distance_b = self.distance_matrix[gene_p - 1][gene_b - 1]

            if distance_a < distance_b:
                child_seq[count] = gene_a
                gene_p = gene_a
                idx_a = (parent1[gene_p] + 1) % self.gene_num
            else:
                child_seq[count] = gene_b
                gene_p = gene_b
                idx_b = (parent2[gene_p] + 1) % self.gene_num

            count += 1

        child.gene_seq = child_seq
        child.compute_distance()

        return child

    def crossover_RX(self, chro, pr):
        """
        Random Crossover
        有助于提高种群的多样性，适合与其它交叉方式结合
        1. 从Parent1中随机选出pr%的城市从头加入子代
        2. 从Parent2中按顺序挑出没有在子代中未出现的城市加入子代
        3. 先将Parent1的基因顺序打乱，然后取前pr * gene_num个基因加入到子代即可
        """

        child = Chromosome(self.gene_num, self.distance_matrix)
        child_seq = []  # 使用list的append比较方便
        temp_seq = self.gene_seq  # Parent1基因序列副本
        gene_num_p1 = int(pr * self.gene_num)  # 取自Parent1的基因数

        for i in range(self.gene_num):
            j = random.randint(0, self.gene_num - 1)
            temp = temp_seq[i]
            temp_seq[i] = temp_seq[j]
            temp_seq[j] = temp

        # 来自Parent1的基因
        for i in range(gene_num_p1):
            child_seq.append(temp_seq[i])

        # 来自Parent2的基因
        for i in range(self.gene_num):
            if chro.gene_seq[i] not in child_seq:
                child_seq.append(chro.gene_seq[i])

        child.gene_seq = child_seq
        child.compute_distance()

        return child

    def mutation(self):
        """
        染色体变异，随机选取两个位点的基因交换
        """
        gene_idx1 = random.randint(0, self.gene_num - 1)
        gene_idx2 = random.randint(0, self.gene_num - 1)
        self.swap_gene(self, gene_idx1, gene_idx2)
        self.compute_distance()  # 重新计算亲和度

    @staticmethod
    def swap_gene(chro, gene_idx1, gene_idx2):
        temp = chro.gene_seq[gene_idx1]
        chro.gene_seq[gene_idx1] = chro.gene_seq[gene_idx2]
        chro.gene_seq[gene_idx2] = temp


class Population:
    def __init__(self,
                 chro_num=0,
                 gene_num=0,
                 distance_matrix=None,
                 crossover_prob=0.8,
                 mutation_prob=0.2,
                 memory_size=10,
                 crossover_mode='OX'):

        self.chro_num = chro_num
        self.gene_num = gene_num
        self.distance_matrix = distance_matrix
        self.crossover_prob = crossover_prob  # 交叉概率
        self.mutation_prob = mutation_prob  # 变异概率
        self.memory_size = memory_size  # 记忆库大小
        self.crossover_mode = crossover_mode  # crossover的方式
        self.memory = []  # 记忆库
        self.generation = 0
        self.chromosomes = []
        self.fitness_probs = []  # 适应度概率
        self.best_generation = 0  # 最好个体所在的generation
        self.best_chromosome = Chromosome(gene_num, distance_matrix)  # 历史最好的个体

    def init_population(self):
        """
        随机初始化种群
        """
        self.generation = 1

        # 生成个体
        for i in range(self.chro_num):
            chromosome = Chromosome(self.gene_num,
                                    self.distance_matrix)
            chromosome.init_chromosome()
            chromosome.compute_distance()

            self.chromosomes.append(chromosome)

        self.chromosomes.sort(key=self.take_distance, reverse=False)  # 按距离从小到大排序
        self.find_best()  # 保存历史最好个体

        # 初始化记忆库
        for i in range(self.memory_size):
            new_chro = Chromosome()
            new_chro.duplication(self.chromosomes[i])
            self.memory.append(new_chro)

        # 初始化适应度概率
        self.compute_fitness_probs()

    def display_chromosomes(self):
        print(f'Generation {self.generation}:')
        for i in range(self.chro_num):
            self.chromosomes[i].display_chromosome()

    def compute_fitness_probs(self):
        """
        计算适应度概率
        """

        min_distance = self.chromosomes[0].get_distance()
        max_distance = self.chromosomes[-1].get_distance()

        for i in range(self.chro_num):
            distance = self.chromosomes[i].get_distance()
            unfitness_prob = (distance - min_distance) / (max_distance - min_distance + 1e-5)
            fitness_prob = 1.0 - unfitness_prob

            # SECTION: sigmoid
            fitness_prob = fitness_prob - 0.5
            fitness_prob = 10 * fitness_prob
            fitness_prob = utils.sigmoid(fitness_prob)

            self.fitness_probs.append(fitness_prob)

    def find_best(self):
        """
        寻找历史最优个体
        """
        # print('cur best distance:', self.chromosomes[0].get_distance())
        # print('history best distance: ', self.best_chromosome.get_distance())
        # print('histroy best seq: ', self.best_chromosome.gene_seq)
        if self.chromosomes[0].get_distance() < self.best_chromosome.get_distance():
            self.best_chromosome.duplication(self.chromosomes[0])
            self.best_generation = self.generation

        # print(f'Chromosome {best_idx + 1} is the best one in generation {self.generation}! '
        #       f'Its gene sequence is: ')
        # self.chromosomes[best_idx].display_chromosome()
        # print()

    def memorization(self):
        """
        将亲和度最好的几个个体存储到记忆库中，用于之后更新种群
        """
        # 当前种群中的个体
        sorted_chromosomes = []
        for i in range(self.chro_num):
            sorted_chromosomes.append(self.chromosomes[i])

        # 记忆库中保存的个体
        for i in range(self.memory_size):
            sorted_chromosomes.append(self.memory[i])

        sorted_chromosomes.sort(key=self.take_distance, reverse=False)

        for i in range(self.memory_size):
            self.memory[i] = sorted_chromosomes[i]

    @staticmethod
    def take_distance(chro):
        return chro.get_distance()

    def selection(self):
        new_chromosomes = []  # 选择后的种群
        for i in range(self.chro_num):
            new_chromosome = Chromosome()
            new_chromosomes.append(new_chromosome)

        count = 0
        while count < self.chro_num:
            rand_idx = random.randint(0, self.chro_num - 1)
            fitness_prob = self.fitness_probs[rand_idx]
            rand_prob = random.random()
            if rand_prob < fitness_prob:
                # 存活
                new_chromosomes[count].duplication(self.chromosomes[rand_idx])
                count += 1
            else:
                pass

        self.chromosomes = new_chromosomes

    def selection_roulette(self):
        """
        种群选择，基于激励度，采用轮盘赌。
        1. 根据每个个体的simulation probability计算累计概率
        2. 从[0, 1)中随机采样一个数来决定哪个个体存活
        3. 共进行chro_num次选择，以保证种群大小不变
        """

        new_chromosomes = []  # 存活个体
        selected = np.zeros(self.chro_num, dtype=np.int32)  # 被选中染色体的索引
        cumulative_probs = np.zeros(self.chro_num)  # 累计概率

        for i in range(self.chro_num):
            new_chromosome = Chromosome()
            new_chromosomes.append(new_chromosome)

        # 计算累计概率
        cumulative_probs[0] = self.fitness_probs[0]
        for i in range(1, self.chro_num):
            cumulative_probs[i] = cumulative_probs[i - 1] + self.fitness_probs[i]

        # 轮盘赌 ROULETTE
        for i in range(self.chro_num):
            rand = random.random()  # [0,1)
            if rand <= cumulative_probs[0]:
                selected[i] = 0
            else:
                # 循环找出rand介于哪两个概率之间
                for j in range(self.chro_num - 1):
                    if cumulative_probs[j] < rand <= cumulative_probs[j + 1]:
                        selected[i] = j

        # 更新种群
        for i in range(self.chro_num):
            new_chromosomes[i].duplication(self.chromosomes[i])
        self.chromosomes = new_chromosomes

    def crossover(self):
        """
        种群中个体之间的交叉
        1. 随机选出两个个体进行交叉
        2. 两个个体依据概率是否交叉
        3. 直到子代个体数达到chro_num
        4. 自己不和自己交叉
        """
        count = 0  # 记录子代个体的数目
        child_chromosomes = []

        while count < self.chro_num:
            rand = random.random()  # 依概率发生交叉
            if rand <= self.crossover_prob:
                chro_idx1 = random.randint(0, self.chro_num - 1)
                chro_idx2 = random.randint(0, self.chro_num - 1)
                # 自己不和自己交叉
                if chro_idx1 == chro_idx2:
                    offset = random.randint(1, self.chro_num - 1)
                    chro_idx2 = (chro_idx1 + offset) % self.chro_num

                chro1 = self.chromosomes[chro_idx1]
                chro2 = self.chromosomes[chro_idx2]

                # 选择crossover的方式
                if self.crossover_mode == 'OX':
                    child, _, _ = chro1.crossover_OX(chro2)
                elif self.crossover_mode == 'MOX':
                    child = chro1.crossover_MOX(chro2)
                elif self.crossover_mode == 'SCX':
                    child = chro1.crossover_SCX(chro2)
                elif self.crossover_mode == 'MSCX':
                    child = chro1.crossover_MSCX(chro2)
                else:
                    raise ValueError('Not supported mode!')

                child_chromosomes.append(child)

                count += 1

        self.chromosomes = child_chromosomes

    def mutation(self):
        """
        种群中个体的变异
        1. 遍历每个个体
        2. 依概率发生变异
        """
        for i in range(self.chro_num):
            rand = random.random()
            if rand <= self.mutation_prob:
                self.chromosomes[i].mutation()

    def update(self):
        """
        种群更新
        1. 记忆库个体加入
        2. 适应度概率
        3. 最好个体
        """
        self.generation += 1

        # 记忆库个体加入
        self.chromosomes.sort(key=self.take_distance, reverse=True)  # 替换距离大的个体
        for i in range(self.memory_size):
            new_chro = Chromosome(self.gene_num, self.distance_matrix)
            new_chro.duplication(self.memory[i])
            self.chromosomes[i] = new_chro

        # 更新适应度概率
        self.compute_fitness_probs()

        self.chromosomes.sort(key=self.take_distance, reverse=False)

        # 更新最好个体
        self.find_best()
