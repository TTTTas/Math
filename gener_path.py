import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from sklearn.cluster import DBSCAN
from tqdm import tqdm


def compute_path_fun_score(path, edge_scores, edge_lengths, feature_dict, weights=(0.7, 0.15, 0.15), plot=False):
    """
    计算路径的游览趣味性指标

    参数:
        path: List[int] 节点序列
        edge_scores: Dict[(i,j), float] 每条边的异景程度
        edge_lengths: Dict[(i,j), float] 每条边长度
        feature_dict: Dict[node_id, Dict[str, np.ndarray]] 每个节点的景观特征向量
        weights: tuple(float) 最终加权权重 (w1, w2, w3)
        plot: bool 是否绘制异景变化与节点类型分布

    返回:
        dict 包含：
            'edge_mean': 加权异景均值
            'edge_std': 异景波动
            'node_div': 节点特征多样性
            'fun_score': 最终加权趣味性分数
    """

    # --- 1. 加权异景均值（长度负相关） ---
    edge_vals = []
    edge_weights = []
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        score = edge_scores.get((i, j), edge_scores.get((j, i), 0))
        length = edge_lengths.get((i, j), edge_lengths.get((j, i), 1.0))
        edge_vals.append(score)
        edge_weights.append(1.0 / (length + 1e-6))  # 长度越大权重越小

    edge_vals = np.array(edge_vals)
    edge_weights = np.array(edge_weights)
    edge_mean = np.sum(edge_vals * edge_weights) / np.sum(edge_weights)

    # --- 2. 异景波动 (加权标准差) ---
    edge_std = np.sqrt(np.sum(edge_weights * (edge_vals - edge_mean) ** 2) / np.sum(edge_weights))

    # --- 3. 节点特征多样性 (Shannon entropy) ---
    node_vecs = []
    for n in path:
        vec = np.zeros(4)
        types = ['building', 'rock', 'plant', 'water']
        for i, t in enumerate(types):
            vec[i] = np.sum(feature_dict[n][t])
        node_vecs.append(vec)
    node_vecs = np.array(node_vecs)
    total_vec = np.sum(node_vecs, axis=0)
    norm_vec = total_vec / (np.sum(total_vec) + 1e-8)
    node_div = -np.sum(norm_vec * np.log(norm_vec + 1e-8))  # Shannon entropy

    # --- 4. 最终加权分数 ---
    w1, w2, w3 = weights
    fun_score = w1 * edge_mean + w2 * edge_std + w3 * node_div

    # --- 可选绘图 ---
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

        # 异景变化
        axs[0].plot(range(len(edge_vals)), edge_vals, '-o', color='red')
        axs[0].set_ylabel("Edge Score")
        axs[0].set_title("Edge Scores along Path")

        # 节点类型分布
        types = ['building', 'rock', 'plant', 'water']
        for i in range(4):
            axs[1].plot(range(len(path)), node_vecs[:,i], '-o', label=types[i])
        axs[1].set_ylabel("Feature Value")
        axs[1].set_xlabel("Node Index along Path")
        axs[1].legend()
        axs[1].set_title("Node Feature Distribution")

        plt.tight_layout()
        plt.show()

    return fun_score


def genetic_path_planning(nodes, edges, edge_scores, edge_lengths, start, end, feature_dict,
                          population_size=50, generations=200, mutation_prob=0.2,
                          alpha=0.6, beta=3, gamma=0.2, delta=0.5,
                          max_path_length=None, cluster_eps=15.0):
    """
    遗传算法规划从 start 到 end 的路径，目标：
        - 最大化路径趣味性
        - 最小化重复节点比例（加权）
        - 最大化覆盖范围
        - 减少同一区域重复访问（节点簇抽稀）

    参数:
        delta: 区域重复惩罚权重
        cluster_eps: DBSCAN 聚类半径
    """
    adjacency = {n: [] for n in nodes}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    all_nodes_set = set(nodes.keys())

    # --- 节点簇聚类 ---
    coords = np.array(list(nodes.values()))
    clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(coords)
    node_cluster = {node_id: label for node_id, label in zip(nodes.keys(), clustering.labels_)}

    # 绘图
    plt.figure(figsize=(8, 8))
    labels = clustering.labels_
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_points = coords[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[colors(i)], label=f"Cluster {label}", s=50, alpha=0.6, edgecolors="k")

    # 标注节点编号
    for node_id, (x, y) in nodes.items():
        plt.text(x, y, str(node_id), fontsize=8, ha="right", va="bottom")

    plt.title("Node Clusters")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.show()

    # --- 初始化种群 ---
    def random_path():
        path = [start]
        visited = {start}
        current = start
        steps = 0
        max_attempts = 1000
        min_steps = 2

        while (current != end or steps < min_steps) and steps < max_attempts:
            neighbors = adjacency.get(current, [])
            if not neighbors:
                break
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            next_node = random.choice(unvisited_neighbors) if unvisited_neighbors else random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            steps += 1
            if max_path_length and len(path) >= max_path_length:
                break

        if start == end and path[-1] != start:
            path.append(start)
        return path

    population = [random_path() for _ in range(population_size)]

    # --- 适应度函数 ---
    def fitness(path):
        # 1. 趣味性
        fun_score = compute_path_fun_score(path, edge_scores, edge_lengths, feature_dict)

        # 2. 重复节点惩罚
        counts = {}
        for node in path:
            counts[node] = counts.get(node, 0) + 1
        weighted_repeat = sum((c - 1) for c in counts.values() if c > 1) / len(path)

        # 3. 区域重复惩罚
        cluster_counts = {}
        for node in path:
            c = node_cluster[node]
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        cluster_penalty = sum(max(0, count - 2) for count in cluster_counts.values()) / len(path)

        # 4. 覆盖率
        coverage = len(set(path)) / len(all_nodes_set)

        return alpha * fun_score - beta * weighted_repeat - delta * cluster_penalty + gamma * coverage

    # --- 选择 ---
    def select_pair(pop):
        weights = np.array([fitness(p) for p in pop])
        weights = np.maximum(weights, 0)
        probs = weights / (weights.sum() + 1e-6)
        return random.choices(pop, probs, k=2)

    # --- 交叉 ---
    def crossover(p1, p2):
        common = list(set(p1) & set(p2))
        if not common:
            return deepcopy(p1), deepcopy(p2)
        cross = random.choice(common)
        i1 = p1.index(cross)
        i2 = p2.index(cross)
        child1 = p1[:i1] + p2[i2:]
        child2 = p2[:i2] + p1[i1:]
        return child1, child2

    # --- 变异 ---
    def mutate(path):
        if len(path) < 2:
            return path
        idx = random.randint(1, len(path) - 2)
        node = path[idx]
        neighbors = adjacency[path[idx - 1]]
        path[idx] = random.choice(neighbors)
        return path

    # --- 主循环 ---
    best_path, best_score = None, -np.inf
    prev_best_score = None

    for gen in tqdm(range(generations)):
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = select_pair(population)
            c1, c2 = crossover(p1, p2)
            if random.random() < mutation_prob:
                c1 = mutate(c1)
            if random.random() < mutation_prob:
                c2 = mutate(c2)
            new_population.extend([c1, c2])
        population = new_population[:population_size]

        # 更新最优
        gen_best_score = -np.inf
        gen_best_path = None
        for p in population:
            f = fitness(p)
            if f > gen_best_score:
                gen_best_score = f
                gen_best_path = p

        # 提前停止条件
        if prev_best_score is not None and abs(gen_best_score - prev_best_score) < 1e-6:
            best_score = gen_best_score
            best_path = gen_best_path
            print(f"提前停止：第 {gen + 1} 代")
            break

        prev_best_score = gen_best_score
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_path = gen_best_path

    return best_path, best_score