import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.linestring import LineString
from shapely.geometry import Point, Polygon
from typing import Callable, Dict,Any

from Rs_type import LandscapeElement, LandscapeCollection


def distance_decay(s: float, alpha: float = 0.1) -> float:
    """指数衰减函数"""
    return math.exp(-alpha * s)

def point_in_element(point: Point, element) -> bool:
    geom = element.geometry
    if geom.type == "Polygon":
        if isinstance(geom.coordinates, dict) and "shell" in geom.coordinates:
            # 水体
            shell = geom.coordinates["shell"]
            holes = geom.coordinates.get("holes", [])
            polygon = Polygon(shell, holes)
        else:
            # 建筑或假山
            polygon = Polygon(geom.coordinates)
        return polygon.contains(point)
    elif geom.type == "Circle":
        cx, cy = geom.center
        r = geom.radius
        return (point.x - cx)**2 + (point.y - cy)**2 <= r**2
    elif geom.type == "Polyline":
        line_coords = geom.coordinates["points"]
        line = Polygon(line_coords)  # 简化处理
        return point.distance(line) < 1e-6
    return False


def exponential_decay(max_distance: float, lam: float = 3.0) -> Callable[[float], float]:
    """
    构造指数衰减函数 w(d) = exp(-lambda * d / max_distance)

    参数：
        max_distance: 射线最大距离，用于归一化
        lam: 衰减速率，lambda 越大远景影响越小，默认 3.0

    返回：
        decay_func(distance) -> 权重
    """

    def decay_func(d: float) -> float:
        if d <= 0:
            return 1.0  # 近处权重最大
        elif d >= max_distance:
            return 0.0  # 超过最大距离权重为0
        else:
            return math.exp(-lam * d / max_distance)

    return decay_func

def construct_sector_feature_vector(
        landscape,
        observation_point: tuple,
        n_sectors: int = 8,
        rays_per_sector: int = 5,
        max_distance: float = 50.0,
        decay_func: Callable[[float], float] = None,
        r_safe: float = 0.5,
        sample_step: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    构造观赏点的空间分布特征向量（8个扇形区域）

    输出：
        Dict[str, np.ndarray]，每个景观类型对应一个长度为 n_sectors 的数组
    """
    if decay_func is None:
        decay_func = lambda d: max(0, 1 - d / max_distance)

    feature_vector = {etype: np.zeros(n_sectors) for etype in ['building', 'rock', 'plant', 'water']}

    ox, oy = observation_point
    sector_angles = np.linspace(0, 2 * np.pi, n_sectors + 1)

    for sector_idx in range(n_sectors):
        start_angle = sector_angles[sector_idx]
        end_angle = sector_angles[sector_idx + 1]
        rays = np.linspace(start_angle, end_angle, rays_per_sector, endpoint=False)

        for angle in rays:
            dx, dy = np.cos(angle), np.sin(angle)
            distance_step = 0.0
            ray_transparency = 1.0  # 初始化射线可视强度
            while distance_step < max_distance:
                distance_step += sample_step
                if distance_step <= r_safe:
                    continue

                px = ox + dx * distance_step
                py = oy + dy * distance_step
                point = Point(px, py)

                blocked = False
                for element in landscape.elements:
                    if point_in_element(point, element):
                        # if element.type == "rock":
                            # print(1)
                        weight = decay_func(distance_step) * element.weight * ray_transparency
                        feature_vector[element.type][sector_idx] += weight
                        # 更新射线通视度
                        ray_transparency *= element.transparent
                        if element.transparent < 1e-5:
                            blocked = True  # 标记射线被阻挡

                if blocked:
                    break  # 射线被阻挡，停止延伸

    return feature_vector


def generate_direction_labels(n_sectors: int) -> list:
    """
    根据扇形数量生成方向标签
    - n_sectors = 8: ['E','NE','N','NW','W','SW','S','SE']
    - n_sectors = 16: ['E','ENE','NE','NNE','N','NNW','NW','WNW','W','WSW','SW','SSW','S','SSE','SE','ESE']
    - n_sectors 可自由扩展
    """
    # 八方位基础标签
    base_8 = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']

    if n_sectors == 8:
        return base_8
    elif n_sectors == 16:
        # 每个八方位细分成两段
        subdiv_16 = []
        for i in range(8):
            curr = base_8[i]
            next_idx = (i + 1) % 8
            next_dir = base_8[next_idx]
            # 两段标签：当前+当前-下一
            subdiv_16.append(curr)
            subdiv_16.append(curr + next_dir)
        return subdiv_16
    else:
        # 通用处理：按八方位插值生成标签
        # 计算每个扇形对应角度
        angles = np.linspace(0, 360, n_sectors, endpoint=False)
        labels = []
        for a in angles:
            # 根据角度映射到八方位
            idx = int((a + 22.5) // 45) % 8
            labels.append(base_8[idx])
        return labels

def plot_sector_feature_vector(feature_vector: dict, n_sectors: int = 8):
    """
    绘制观测点扇形特征向量雷达图：
    - 第0扇形对应正东方向
    - 逆时针增加
    - 显示4种景观类型
    """
    types = list(feature_vector.keys())

    # 扇形角度闭合
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 方向标签（逆时针对应角度）
    direction_labels = generate_direction_labels(n_sectors)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {
        'water': 'blue',
        'building': 'red',
        'rock': 'brown',
        'plant': 'green'
    }

    for t in types:
        values = feature_vector[t].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, color=colors.get(t, 'gray'), linewidth=2, label=t)
        ax.fill(angles, values, color=colors.get(t, 'gray'), alpha=0.25)

    # 设置雷达图 0° 对应正东，逆时针增加
    ax.set_theta_zero_location('E')  # 0° → 东
    ax.set_theta_direction(1)  # 逆时针增加

    # 标注方向
    tick_angles = np.array(angles[:-1]) # 每个扇形
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(direction_labels)

    ax.set_rlabel_position(225)
    ax.set_title("观测点扇形特征向量", fontsize=14)
    ax.legend(loc='upper right')
    # plt.show()


# ==============异景程度计算=============== #
EPS = 1e-12

def flatten_feature_dict(fv_point: Dict[str, np.ndarray]) -> np.ndarray:
    """把类型->(S,) 的字典拼成一个一维向量（固定顺序）"""
    keys = sorted(fv_point.keys())
    return np.concatenate([fv_point[k].ravel() for k in keys])

def compute_S_and_flat_features(feature_dict: Dict[Any, Dict[str, np.ndarray]]) -> Tuple[
    Dict[Any, float], Dict[Any, np.ndarray]]:
    """
    输入: feature_dict: {node: {type: np.array(S,)}}
    返回:
        S_dict: {node: total_sum}
        flat: {node: flattened vector F_p}
    """
    S = {}
    flat = {}
    for p, fv in feature_dict.items():
        vec = flatten_feature_dict(fv)
        flat[p] = vec
        S[p] = float(np.sum(vec))
    return S, flat

def normalize_pointwise_flat(flat_vec: np.ndarray, total: float, eps: float = EPS) -> np.ndarray:
    return flat_vec / (total + eps)

def magnitude_normalize_whole(S_dict: Dict[Any, float], quantile: float = 0.95) -> Dict[Any, float]:
    """
    用全图分位数截断再线性缩放到 [0,1]
    quantile: e.g., 0.95 表示以95%分位为上限
    """
    vals = np.array(list(S_dict.values()), dtype=float)
    if vals.size == 0:
        return {k: 0.0 for k in S_dict}
    Q = float(np.quantile(vals, quantile))
    if Q <= 0:
        Q = float(np.max(vals)) + EPS
    out = {}
    for k, v in S_dict.items():
        out[k] = float(np.clip(v / (Q + EPS), 0.0, 1.0))
    return out

def compute_edge_hybrid_scores(
        feature_dict: Dict[Any, Dict[str, np.ndarray]],
        edges: list,
        pattern_metric: str = "l2",  # "l2" or "l1"
        quantile_clip: float = 0.95,
        alpha: float = 1.0,
        beta: float = None
) -> Tuple[Dict[tuple, float], Dict[Any, float]]:
    """
    计算边的异景程度，混合 pattern 与 magnitude。
    返回 (edge_scores, magnitude_normalized_dict)
    参数:
      - feature_dict: {node: {type: np.array(S,)}}
      - edges: list of (u,v)
      - pattern_metric: "l2" or "l1"
      - quantile_clip: 用于 magnitude 归一化的上限分位数
      - alpha: multiplier in multiplicative combo D = P * (1 + alpha * avg_mag)
      - beta: 如果指定，就使用线性混合 D = beta*P + (1-beta)*avg_mag
    """
    # 1) flatten and totals
    S_dict, flat = compute_S_and_flat_features(feature_dict)
    # 2) magnitude normalized by quantile
    mag_norm = magnitude_normalize_whole(S_dict, quantile=quantile_clip)
    # 3) pattern vectors (pointwise normalization)
    pattern = {}
    for p in flat:
        pattern[p] = normalize_pointwise_flat(flat[p], S_dict[p])

    # 4) compute edge scores
    edge_scores = {}
    for (u, v) in edges:
        if u not in pattern or v not in pattern:
            edge_scores[(u, v)] = 0.0
            continue
        a = pattern[u]
        b = pattern[v]
        if pattern_metric == "l2":
            P = float(np.linalg.norm(a - b, ord=2))
        else:
            P = float(np.linalg.norm(a - b, ord=1))
        avg_mag = 0.5 * (mag_norm.get(u, 0.0) + mag_norm.get(v, 0.0))
        if beta is None:
            D = P * (1.0 + alpha * avg_mag)
        else:
            D = float(beta * P + (1.0 - beta) * avg_mag)
        edge_scores[(u, v)] = D
    return edge_scores, mag_norm