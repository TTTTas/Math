from dataclasses import dataclass, field
from typing import Dict
from typing import List, Tuple, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix


# 定义几何类型
@dataclass
class Geometry:
    type: str  # "Polygon", "Circle", "Polyline"

    # Polygon: {"shell": [...], "holes": [[...], [...]]}
    # Polyline: {"points": [...]}
    # Circle: 不使用 coordinates
    coordinates: Optional[Union[List[Tuple[float, float]], Dict[str, List[List[Tuple[float, float]]]]]] = None

    center: Optional[Tuple[float, float]] = None  # Circle
    radius: Optional[float] = None  # Circle 半径

# 景观元素
@dataclass
class LandscapeElement:
    id: int
    type: str  # "rock", "building", "water", "plant"
    geometry: Geometry
    transparent: float
    weight: float
    name: Optional[str] = None
    cluster_id: Optional[int] = None

# 景观集合容器
@dataclass
class LandscapeCollection:
    elements: List[LandscapeElement] = field(default_factory=list)

    def add_element(self, element: LandscapeElement):
        self.elements.append(element)

    def get_by_type(self, element_type: str) -> List[LandscapeElement]:
        return [e for e in self.elements if e.type == element_type]

    def __len__(self):
        return len(self.elements)


def compute_landscape_features_extended(landscape: LandscapeCollection) -> Dict[str, float]:
    """
    对 LandscapeCollection 计算结构特征，包括：
    1. 每类元素数量、总面积（归一化/对数）、平均面积、面积比例
    2. 最近邻距离均值与标准差
    3. 不同类型元素间平均距离（如水体-建筑）
    4. 整体离散度指标
    """
    features = {}
    element_types = ["water", "rock", "plant", "building"]

    # --- 每类元素特征 ---
    type_areas = {}
    type_counts = {}
    type_centers = {}

    total_area_all = 0
    for elem_type in element_types:
        elems = landscape.get_by_type(elem_type)
        type_counts[elem_type] = len(elems)

        areas = []
        centers = []
        for e in elems:
            geom = e.geometry
            area = 0
            if geom.type == "Polygon":
                coords = geom.coordinates["shell"] if isinstance(geom.coordinates, dict) else geom.coordinates
                if coords:
                    xs, ys = zip(*coords)
                    area += 0.5 * abs(sum(xs[i]*ys[i+1]-xs[i+1]*ys[i] for i in range(len(xs)-1)))
            elif geom.type == "Circle" and geom.radius is not None:
                area += np.pi * geom.radius**2
            areas.append(area)

            # 中心点
            if geom.center:
                centers.append(geom.center)
            elif geom.type == "Polygon" and coords:
                centers.append((np.mean(xs), np.mean(ys)))

        type_areas[elem_type] = areas
        type_centers[elem_type] = centers

        features[f"{elem_type}_count"] = len(elems)

        total_area = sum(areas)
        total_area_all += total_area

        # 对数缩放，减小大数值影响
        features[f"{elem_type}_total_area"] = 1 * np.log1p(total_area)

        # 平均面积（同样缩放处理，避免异常值）
        features[f"{elem_type}_avg_area"] = 1 * np.log1p(np.mean(areas)) if areas else 0

        # 最近邻距离
        if len(centers) > 1:
            coords_arr = np.array(centers)
            dmat = distance_matrix(coords_arr, coords_arr)
            np.fill_diagonal(dmat, np.inf)
            nn_dist = dmat.min(axis=1)
            features[f"{elem_type}_mean_nn_dist"] = nn_dist.mean()
            features[f"{elem_type}_std_nn_dist"] = nn_dist.std()
        else:
            features[f"{elem_type}_mean_nn_dist"] = 0
            features[f"{elem_type}_std_nn_dist"] = 0

    # --- 总面积比例 ---
    for elem_type in element_types:
        total_area = sum(type_areas[elem_type])
        features[f"{elem_type}_area_ratio"] = total_area / total_area_all if total_area_all > 0 else 0

    # --- 不同类型元素间平均距离 ---
    for i, type_a in enumerate(element_types):
        for type_b in element_types[i+1:]:
            centers_a = type_centers.get(type_a, [])
            centers_b = type_centers.get(type_b, [])
            if centers_a and centers_b:
                dmat = distance_matrix(np.array(centers_a), np.array(centers_b))
                features[f"{type_a}_to_{type_b}_mean_dist"] = dmat.mean()
                features[f"{type_a}_to_{type_b}_std_dist"] = dmat.std()
            else:
                features[f"{type_a}_to_{type_b}_mean_dist"] = 0
                features[f"{type_a}_to_{type_b}_std_dist"] = 0

    # --- 整体离散度 ---
    all_centers = [c for centers in type_centers.values() for c in centers]
    if len(all_centers) > 1:
        coords_arr = np.array(all_centers)
        dmat = distance_matrix(coords_arr, coords_arr)
        np.fill_diagonal(dmat, np.inf)
        nn_dist_all = dmat.min(axis=1)
        features["all_elements_mean_nn_dist"] = nn_dist_all.mean()
        features["all_elements_std_nn_dist"] = nn_dist_all.std()
    else:
        features["all_elements_mean_nn_dist"] = 0
        features["all_elements_std_nn_dist"] = 0

    return features


def visualize_landscape_features(features: dict, figsize=(12, 6), max_bars=50):
    """
    可视化景观特征向量（字典形式）

    参数:
        features: dict, 特征名称 -> 特征值
        figsize: 图像尺寸
        max_bars: 最大显示柱数，避免特征过多时拥挤
    """
    # 按 key 排序
    feature_names = list(features.keys())
    feature_values = [features[k] for k in feature_names]

    # 如果特征太多，显示前 max_bars 个
    if len(feature_names) > max_bars:
        feature_names = feature_names[:max_bars]
        feature_values = feature_values[:max_bars]

    x = np.arange(len(feature_names))
    plt.figure(figsize=figsize)
    bars = plt.bar(x, feature_values, color='skyblue', edgecolor='black')
    plt.xticks(x, feature_names, rotation=90, fontsize=8)
    plt.ylabel("Feature Value")
    plt.title("Landscape Feature Vector")
    plt.tight_layout()
    plt.show()