import math
from matplotlib import cm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import heapq
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Set
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

    # --- 5. tanh 压缩映射到 [0,1] ---
    fun_score = np.tanh(fun_score / 80)
    fun_score = (fun_score + 1) / 2  # 映射到 [0,1]

    # --- 可选绘图 ---
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # 异景变化
        axs[0].plot(range(len(edge_vals)), edge_vals, '-o', color='red')
        axs[0].set_ylabel("Edge Score")
        axs[0].set_title("Edge Scores along Path")

        # 节点类型分布
        types = ['building', 'rock', 'plant', 'water']
        for i in range(4):
            axs[1].plot(range(len(path)), node_vecs[:, i], '-o', label=types[i])
        axs[1].set_ylabel("Feature Value")
        axs[1].set_xlabel("Node Index along Path")
        axs[1].legend()
        axs[1].set_title("Node Feature Distribution")

        plt.tight_layout()
        plt.show()

    return fun_score


def _edge_key(u, v):
    return (u, v)


def get_edge_length(edge_lengths: Dict[Tuple[int, int], float], u: int, v: int, default: float = 1.0) -> float:
    return edge_lengths.get((u, v), edge_lengths.get((v, u), default))


def shortest_path(adjacency: Dict[int, List[int]],
                  start: int, end: int,
                  edge_lengths: Optional[Dict[Tuple[int, int], float]] = None) -> Optional[List[int]]:
    """
    返回 start->end 的最短路径（如果 edge_lengths 给出，则用 Dijkstra 否则用 BFS）。
    如果不可达，返回 None。
    """
    if start == end:
        return [start]

    if edge_lengths is None:
        # 无权 BFS
        q = deque([[start]])
        seen = {start}
        while q:
            path = q.popleft()
            node = path[-1]
            for nb in adjacency.get(node, []):
                if nb in seen:
                    continue
                if nb == end:
                    return path + [nb]
                seen.add(nb)
                q.append(path + [nb])
        return None
    else:
        # Dijkstra
        pq = [(0.0, start, [start])]
        dist = {start: 0.0}
        while pq:
            d, node, path = heapq.heappop(pq)
            if node == end:
                return path
            if d > dist.get(node, float('inf')):
                continue
            for nb in adjacency.get(node, []):
                nd = d + get_edge_length(edge_lengths, node, nb, default=1.0)
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb, path + [nb]))
        return None


def generate_candidate_paths(nodes, edges, edge_scores, start, end,
                             score_percentile=0.9, k_paths=10, max_length=None):
    """
    生成候选路径集合：
    1. 筛选异景程度较高的边作为关键边
    2. 将关键边端点作为候选集
    3. 在候选点之间用 K 最短路径算法找路径
    4. 补充一些随机游走，增加多样性

    参数:
        nodes: dict[node_id] = (x,y)
        edges: list of (i,j)
        edge_scores: dict[(i,j)] = score
        start, end: 指定起终点
        score_percentile: 取多少百分位以上的边作为关键边
        k_paths: 每对候选点之间取多少条最短路径
        max_length: 最大路径长度（可选）

    返回:
        paths: list[list[node]]
    """
    # --- 构造图 ---
    G = nx.Graph()
    for i, j in edges:
        length = ((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2) ** 0.5
        G.add_edge(i, j, weight=length)

    # --- 1. 选出高分边 ---
    scores = list(edge_scores.values())
    if not scores:
        return []
    threshold = np.percentile(scores, score_percentile * 100)
    candidate_edges = [e for e, s in edge_scores.items() if s >= threshold]

    # --- 2. 候选点集 ---
    candidate_nodes = set()
    for u, v in candidate_edges:
        candidate_nodes.add(u)
        candidate_nodes.add(v)

    # 确保起点终点包含
    candidate_nodes.add(start)
    candidate_nodes.add(end)
    candidate_nodes = list(candidate_nodes)

    # --- 3. 在候选点对之间找 K 最短路径 ---
    paths = []
    for u in candidate_nodes:
        for v in candidate_nodes:
            if u == v:
                continue
            try:
                gen = nx.shortest_simple_paths(G, u, v, weight="weight")
                for k, p in zip(range(k_paths), gen):
                    if max_length and len(p) > max_length:
                        continue
                    paths.append(p)
            except nx.NetworkXNoPath:
                continue

    # --- 4. 补充随机游走 ---
    def random_walk(start, steps=10):
        path = [start]
        current = start
        for _ in range(steps):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            path.append(current)
        return path

    for _ in range(k_paths):
        paths.append(random_walk(start, steps=random.randint(5, 15)))

    # --- 5. 确保包含起点终点 ---
    valid_paths = [p for p in paths if p[0] == start and p[-1] == end]
    if not valid_paths:
        try:
            sp = nx.shortest_path(G, start, end, weight="weight")
            valid_paths.append(sp)
        except nx.NetworkXNoPath:
            pass

    return valid_paths


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    uni = len(set_a | set_b)
    return inter / uni if uni > 0 else 0.0


def generate_initial_path(adjacency, nodes_coords, edges, edge_scores, edge_lengths,
                          start, end, k_paths=50, max_length=40, score_percentile=0.9):
    """
    生成初始路径（保证起终点唯一、候选路径多样化）：
    - 对 start==end 的情况，构造环
    - 避免路径中重复出现起点，除闭环最后回到起点
    """
    import random

    def next_node_choice(current, visited):
        neighbors = adjacency[current]
        if not neighbors:
            return None

        scores = [edge_scores.get((current, n), edge_scores.get((n, current), 0.1)) for n in neighbors]
        total = sum(scores)
        probs = [s / total for s in scores]
        # 优先未访问节点
        unvisited = [n for n in neighbors if n not in visited]
        if unvisited:
            unvisited_probs = [probs[neighbors.index(n)] for n in unvisited]
            total_un = sum(unvisited_probs)
            unvisited_probs = [p / total_un for p in unvisited_probs]

            return random.choices(unvisited, weights=unvisited_probs, k=1)[0]
        else:
            return random.choices(neighbors, weights=probs, k=1)[0]

    if start != end:
        coords_items = list(nodes_coords.items())

        # 找一个 farthest 节点（可连通）作为随机游走目标
        farthest = None
        for nid, (x, y) in sorted(coords_items, key=lambda x: (x[1][0] - nodes_coords[start][0]) ** 2 +
                                                              (x[1][1] - nodes_coords[start][1]) ** 2, reverse=True):
            if nid == start or nid == end:
                continue
            sp_start = shortest_path(adjacency, start, nid, edge_lengths)
            sp_end = shortest_path(adjacency, nid, end, edge_lengths)
            if sp_start and sp_end:
                farthest = nid
                break
        if farthest is None:
            # 如果没有中间点可选，则直接使用最短路径
            sp = shortest_path(adjacency, start, end, edge_lengths)
            return sp if sp else [start, end]

        # 随机游走生成路径函数
        def random_walk_path(start_node, end_node, max_steps=100):
            path = [start_node]
            visited = {start_node}
            current = start_node
            steps = 0
            while current != end_node and steps < max_steps:
                neighbors = [n for n in adjacency[current] if n not in visited]
                if not neighbors:
                    neighbors = adjacency[current]
                next_node = random.choice(neighbors)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                steps += 1
            return path

        # 生成多条候选路径 start->farthest->end
        k_paths = 20
        paths = []
        attempts = 0
        while len(paths) < k_paths and attempts < k_paths * 20:
            attempts += 1
            path1 = random_walk_path(start, farthest)
            path2 = random_walk_path(farthest, end)
            full_path = path1 + path2[1:]  # 去掉重复中间节点
            if full_path not in paths:
                paths.append(full_path)

        # 必须至少有两条路径才能拼接
        if len(paths) < 2:
            raise ValueError(f"候选路径不足两条，无法生成完整路径: len(paths)={len(paths)}")

        # 选择交集最小的两条路径
        path_sets = [set(p) for p in paths]
        best_pair = (0, 1)
        best_score = None
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = path_sets[i], path_sets[j]
                inter = len(s1 & s2)
                jac = jaccard_similarity(s1, s2)
                score = (inter, jac)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (i, j)

        p1, p2 = paths[best_pair[0]], paths[best_pair[1]]
        p2_rev = list(reversed(p2))
        merged = p1 + [n for n in p2_rev[1:] if n != start and n != end]
        merged.append(end)  # 确保终点
        return merged

    # start == end 构造环
    coords_items = list(nodes_coords.items())
    sx, sy = nodes_coords[start]

    # 找 farthest 可连通节点
    farthest = None
    for nid, (x, y) in sorted(coords_items, key=lambda x: (x[1][0] - sx) ** 2 + (x[1][1] - sy) ** 2, reverse=True):
        if nid == start:
            continue
        sp = shortest_path(adjacency, start, nid, edge_lengths)
        if sp:
            farthest = nid
            break
    if farthest is None:
        return [start]

    # 生成候选路径 start->farthest，排除 start 重复
    paths = []
    attempts = 0
    while len(paths) < k_paths and attempts < k_paths * 20:
        attempts += 1
        path = [start]
        visited = {start}
        current = start
        while current != farthest and len(path) < max_length:
            n = next_node_choice(current, visited)
            if n is None or n == start:
                break
            path.append(n)
            visited.add(n)
            current = n
        if current == farthest and path not in paths:
            paths.append(path)

    # 必须至少有两条路径才能拼接
    if len(paths) < 2:
        raise ValueError(f"候选路径不足两条，无法生成闭环: len(paths)={len(paths)}")

    # --- 两条交集最小路径 ---
    path_sets = [set(p) for p in paths]
    best_pair = (0, 1)
    best_score = None
    n = len(paths)

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = path_sets[i], path_sets[j]
            inter = len(s1 & s2)
            jac = jaccard_similarity(s1, s2)
            score = (inter, jac)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (i, j)

    # 获取最优路径对
    p1, p2 = paths[best_pair[0]], paths[best_pair[1]]

    # 拼接形成环路，避免中间重复 start
    p2_rev = list(reversed(p2))
    merged = p1 + [n for n in p2_rev[1:] if n != start]
    merged.append(start)  # 最后闭环回到起点
    return merged


def genetic_path_planning(nodes, edges, edge_scores, edge_lengths, start, end, feature_dict,
                          population_size=50, generations=200, mutation_prob=0.2,
                          alpha=0.6, beta=1.0, gamma=0.4, delta=0.02,
                          max_path_length=None, cluster_eps=10.0):
    """
    遗传算法路径规划，保证起终点，考虑边重复、点重复、区域重复、趣味性、覆盖率。
    """
    import random
    import numpy as np
    from copy import deepcopy
    from tqdm import tqdm
    from sklearn.cluster import DBSCAN

    # --- 构建邻接表 ---
    adjacency = {n: [] for n in nodes}
    edge_set = set()
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
        edge_set.add(tuple(sorted((i, j))))

    all_nodes_set = set(nodes.keys())

    # --- 节点簇聚类 ---
    coords = np.array(list(nodes.values()))
    clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(coords)
    node_cluster = {node_id: label for node_id, label in zip(nodes.keys(), clustering.labels_)}
    labels = np.array([node_cluster[nid] for nid in nodes.keys()])

    # 获取簇的数量
    n_clusters = len(set(labels))
    colors = cm.get_cmap('tab20', n_clusters)  # 使用tab20调色板

    plt.figure(figsize=(8, 8))
    for idx, (x, y) in enumerate(coords):
        cluster_id = labels[idx]
        plt.scatter(x, y, color=colors(cluster_id), label=f'Cluster {cluster_id}' if idx == 0 else "", s=50)

    # 可选：绘制节点编号
    for nid, (x, y) in nodes.items():
        plt.text(x, y, str(nid), fontsize=8, ha='right', va='bottom')

    plt.title(f"DBSCAN Clustering (eps={cluster_eps})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    # plt.show()

    # --- 初始化种群 ---
    # print("=== 初始化种群 ===")
    population = [generate_initial_path(adjacency, nodes, edges, edge_scores, edge_lengths, start, end)
                  for _ in tqdm(range(population_size), desc="初始化种群")]
    # for idx, path in enumerate(population):
    #     print(f"个体 {idx+1}: {path}")

    # --- 预计算区域趣味性 ---
    cluster_fun_score = {}
    for c in set(node_cluster.values()):
        nodes_in_cluster = [n for n, cl in node_cluster.items() if cl == c]
        scores = []
        for i in range(len(nodes_in_cluster)):
            for j in range(i + 1, len(nodes_in_cluster)):
                p = shortest_path(adjacency, nodes_in_cluster[i], nodes_in_cluster[j], edge_lengths)
                if p:
                    scores.append(compute_path_fun_score(p, edge_scores, edge_lengths, feature_dict))
        if scores:
            cluster_fun_score[c] = np.mean(scores)
        else:
            cluster_fun_score[c] = 0.0

    # --- 适应度函数 ---
    def fitness(path):
        fun_score = compute_path_fun_score(path, edge_scores, edge_lengths, feature_dict)

        # --- 边重复惩罚 ---
        counts_edge = {}
        edge_penalty = 0.0
        for k in range(len(path) - 1):
            e = tuple(sorted((path[k], path[k + 1])))
            counts_edge[e] = counts_edge.get(e, 0) + 1
            if counts_edge[e] > 1:
                edge_penalty += delta * (counts_edge[e] - 1)  # 改为每条重复边惩罚相同

        # 归一化：平均到边数
        edge_penalty /= max(len(path) - 1, 1)

        # --- 点重复惩罚 ---
        counts_node = {}
        node_penalty = 0.0
        for n in path:
            counts_node[n] = counts_node.get(n, 0) + 1
            if counts_node[n] > 1:
                node_penalty += delta * (counts_node[n] - 1)

        # 归一化：平均到节点数
        node_penalty /= max(len(path), 1)

        # --- 区域重复惩罚 ---
        cluster_counts = {}
        cluster_penalty = 0.0
        for n in path:
            c = node_cluster[n]
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
            if cluster_counts[c] > 1:
                cluster_penalty += delta * (cluster_counts[c] - 1)

        # 归一化：平均到涉及的簇数
        cluster_penalty * max(len(set(node_cluster[n] for n in path)), 1)

        # --- 区域未覆盖惩罚 ---
        path_clusters = set(node_cluster[n] for n in path)
        missing_clusters = set(cluster_fun_score.keys()) - path_clusters
        missing_penalty = 0.0
        if missing_clusters:
            # 平均化：每个未覆盖簇贡献平均值
            missing_penalty = sum(cluster_fun_score[c] for c in missing_clusters) / max(len(missing_clusters), 1)

        # --- 覆盖率 ---
        coverage = len(set(path)) / len(all_nodes_set)

        # --- 大圈惩罚 ---
        loop_penalty = 0.0
        near_dist = 8
        far_dist = 15.0
        for i in range(len(path) - 2):
            a, b, c = path[i], path[i + 1], path[i + 2]
            xa, ya = nodes[a]
            xb, yb = nodes[b]
            xc, yc = nodes[c]

            dist_ab = np.linalg.norm(np.array([xa, ya]) - np.array([xb, yb]))
            dist_ac = np.linalg.norm(np.array([xa, ya]) - np.array([xc, yc]))

            if dist_ac < near_dist and dist_ab > far_dist:
                loop_penalty += delta * (dist_ab - far_dist + 1)

        # 归一化：按路径长度
        loop_penalty /= max(len(path) - 2, 1)

        # 最小节点数惩罚
        min_node_ratio = 0.3  # 最少30%的节点
        num_nodes = len(set(path))
        min_nodes_required = int(min_node_ratio * len(all_nodes_set))
        if num_nodes < min_nodes_required:
            min_nodes_penalty = delta * (min_nodes_required - num_nodes)
        else:
            min_nodes_penalty = 0.0

        scores = alpha * fun_score - beta * (edge_penalty + node_penalty + 0.8 * cluster_penalty
                + 2 * missing_penalty + 0.4 * loop_penalty + 0.4 * min_nodes_penalty) + gamma * coverage
        if scores < 0:
            return 1e-6
        else:
            return scores

            # --- 选择 ---

    def select_pair(pop):
        weights = np.array([fitness(p) for p in pop])
        weights = np.maximum(weights, 0)
        probs = weights / (weights.sum() + 1e-6)
        return random.choices(pop, probs, k=2)

    # --- 交叉函数（保证终点不变且合法） ---
    def crossover(p1, p2):
        """
        交叉两条路径，保证终点是 end，路径每条边合法
        """
        common = list(set(p1) & set(p2))
        if not common:
            # 如果没有交点，直接返回原路径，但保证终点合法
            return ensure_path_ends(p1), ensure_path_ends(p2)

        # 随机选择一个交点
        cross = random.choice(common)
        i1, i2 = p1.index(cross), p2.index(cross)

        # child1 = p1 前半 + p2 后半（合法拼接）
        child1 = p1[:i1 + 1]
        for node in p2[i2 + 1:]:
            if tuple(sorted((child1[-1], node))) in edge_set:
                child1.append(node)
            else:
                break

        # child2 = p2 前半 + p1 后半（合法拼接）
        child2 = p2[:i2 + 1]
        for node in p1[i1 + 1:]:
            if tuple(sorted((child2[-1], node))) in edge_set:
                child2.append(node)
            else:
                break

        # 确保终点合法，如果末尾不是 end，用最短路径补全
        child1 = ensure_path_ends(child1)
        child2 = ensure_path_ends(child2)

        return child1, child2

    def ensure_path_ends(path):
        """
        确保路径末尾是终点，且边合法
        """
        if path[-1] == end:
            return path

        # 尝试找到最短路径补全到终点
        sp = shortest_path(adjacency, path[-1], end, edge_lengths)
        if sp:
            # 去掉重复起点节点
            path += sp[1:]
            return path
        else:
            # 无法连通，直接返回 None，遗传算法里可丢弃
            return None

    # --- 变异函数（保证终点不变） ---
    def mutate(path):
        """
        随机改变路径中间节点，保证终点不变
        """
        if len(path) < 3:
            return path

        idx = random.randint(1, len(path) - 2)  # 不变动起点和终点
        prev_node = path[idx - 1]
        next_node = path[idx + 1]

        # 邻居必须合法连接前后节点
        candidates = [n for n in adjacency[prev_node] if n != next_node and tuple(sorted((prev_node, n))) in edge_set
                      and tuple(sorted((n, next_node))) in edge_set]

        if candidates:
            path[idx] = random.choice(candidates)

        return path

    # --- 主循环 ---
    best_path, best_score = None, -np.inf
    prev_best_score = None

    for gen in tqdm(range(generations), desc="遗传算法迭代"):
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
        gen_best_score, gen_best_path = -np.inf, None
        for p in population:
            f = fitness(p)
            if f > gen_best_score:
                gen_best_score = f
                gen_best_path = p

        # 提前停止
        if prev_best_score is not None and abs(gen_best_score - prev_best_score) < 1e-8 and abs(best_score) > 0.1:
            best_path, best_score = gen_best_path, gen_best_score
            break
        prev_best_score = gen_best_score
        if gen_best_score > best_score:
            best_path, best_score = gen_best_path, gen_best_score

        if best_score < 1e-2:
            raise ("得分过小")

    return best_path, best_score
