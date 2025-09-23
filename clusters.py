import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.spatial import ConvexHull, QhullError
from scipy.cluster.hierarchy import linkage, fcluster
from shapely.geometry.multipolygon import MultiPolygon
from sklearn.cluster import OPTICS, DBSCAN
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def cluster_rock(paths, distance_threshold=5.0):
    """
    基于路径点之间的最小距离进行聚类，将可能属于同一实体的路径聚为一类。

    参数：
        paths: list of dict，每个 dict 包含 'items': [("l",(x0,y0),(x1,y1)), ...]
        distance_threshold: float，路径之间最小点距小于该值则认为属于同一类

    返回：
        clusters: list of lists，每个子列表是属于同一类的路径索引
    """
    n = len(paths)
    if n == 0:
        return []

    # 构建邻接表（哪些路径需要合并）
    adjacency = [set() for _ in range(n)]

    # 计算路径点集合
    path_points = []
    for path in paths:
        pts = []
        for item in path['items']:
            if item[0] == 'l':
                pts.append(item[1])
                pts.append(item[2])
        path_points.append(np.array(pts))

    # 检查每对路径的最小点距
    for i in range(n):
        for j in range(i + 1, n):
            pts_i = path_points[i]
            pts_j = path_points[j]
            dists = cdist(pts_i, pts_j)  # 点对点距离
            min_dist = dists.min()
            if min_dist <= distance_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # 使用 DFS 找连通分量
    visited = [False] * n
    clusters = []

    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, cluster)

    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    return clusters


def get_outer_boundary(points):
    """
    给定一簇点，返回最外围的点（按凸包边界顺序排列）。
    针对点数小于3或共线的情况做特殊处理。
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points
    if len(points) == 2:
        return points  # 两点无法构成凸包，直接返回

    pts = np.array(points)

    # 检查点是否几乎共线：x 或 y 范围小于一个阈值
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    eps = 1e-5
    if x_range < eps or y_range < eps:
        # 共线或几乎重合，直接返回排序后的点
        # 按 x 排序，如果 x 相同再按 y 排序
        pts_sorted = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        return pts_sorted.tolist()

    # 正常凸包
    try:
        hull = ConvexHull(pts)
        hull_points = pts[hull.vertices]
        return hull_points.tolist()
    except Exception:
        # 出现精度问题，退化处理：按 x 排序
        pts_sorted = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        return pts_sorted.tolist()


from scipy.spatial.distance import euclidean
import numpy as np


def cluster_plant_circles(circle_paths, distance_threshold=5.0, overlap_ratio_threshold=0.5):
    """
    对植被圆进行聚类，考虑中心距离和面积重叠。

    参数：
        circle_paths: list of dict, 每个 dict 包含 'center'=(x,y) 和 'radius'
        distance_threshold: float, 中心距离小于该值则可能同一簇
        overlap_ratio_threshold: float, 面积重叠比例大于该值也算同一簇

    返回：
        clusters: list of lists，每个子列表包含属于同一簇的圆索引
    """
    n = len(circle_paths)
    if n == 0:
        return []

    # 邻接表
    adjacency = [set() for _ in range(n)]

    def circle_overlap(c1, r1, c2, r2):
        """计算两圆的重叠比例"""
        d = euclidean(c1, c2)
        if d >= r1 + r2:
            return 0.0  # 不相交
        if d <= abs(r1 - r2):
            return 1.0  # 一个圆完全在另一个圆内
        # 两圆部分相交，计算面积公式
        r1_sq, r2_sq = r1 ** 2, r2 ** 2
        part1 = r1_sq * np.arccos((d ** 2 + r1_sq - r2_sq) / (2 * d * r1))
        part2 = r2_sq * np.arccos((d ** 2 + r2_sq - r1_sq) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        intersection = part1 + part2 - part3
        # 返回较小圆面积的重叠比例
        min_area = np.pi * min(r1_sq, r2_sq)
        return intersection / min_area

    # 构建邻接表
    for i in range(n):
        c1, r1 = circle_paths[i]['center'], circle_paths[i]['radius']
        for j in range(i + 1, n):
            c2, r2 = circle_paths[j]['center'], circle_paths[j]['radius']
            d = euclidean(c1, c2)
            overlap_ratio = circle_overlap(c1, r1, c2, r2)
            if d <= distance_threshold or overlap_ratio >= overlap_ratio_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # DFS 找连通分量
    visited = [False] * n
    clusters = []

    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, cluster)

    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    return clusters

def density_cluster_plant(circle_paths, eps=2.0, min_samples=2, radius_weight=1.0):
    """
    对植被（圆）进行密度自适应聚类（DBSCAN），考虑圆心空间距离和半径。

    参数：
        circle_paths: list of dict，每个 dict 包含 'center'=(x,y) 和 'radius'
        eps: float，DBSCAN 的距离阈值（单位同 DXF 坐标）
        min_samples: int，DBSCAN 的最小样本数
        radius_weight: float，将半径加权到距离中，避免密集大植被过度合并

    返回：
        clusters: list of lists，每个子列表是属于同一簇的索引
    """
    if not circle_paths:
        return []

    # 提取圆心和半径
    points = np.array([[p['center'][0], p['center'][1], p['radius'] * radius_weight] for p in circle_paths])

    # DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(points)

    # 构建 clusters，每个簇存索引
    clusters = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # -1 是噪声点，单独处理或忽略
            continue
        idxs = np.where(labels == label)[0].tolist()
        clusters.append(idxs)

    return clusters

def hierarchical_cluster_plant(circle_paths, distance_threshold=5.0):
    """
    对植被（圆）进行层次聚类，考虑圆心空间距离和半径。

    参数：
        circle_paths: list of dict，每个 dict 包含 'center' 和 'radius'
        distance_threshold: float，聚类距离阈值（单位同DXF坐标，m）

    返回：
        clusters: list of lists，每个子列表是属于同一簇的索引
    """
    if not circle_paths:
        return []

    # 提取圆心和半径
    points = np.array([[p['center'][0], p['center'][1], p['radius']] for p in circle_paths])

    # 可以对半径加权，权重 alpha，可调
    alpha = 1.0
    points_scaled = points.copy()
    points_scaled[:, 2] *= alpha  # 半径加权

    # 层次聚类
    Z = linkage(points_scaled, method='single', metric='euclidean')

    # 根据距离阈值划分簇
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # 构建 clusters，每个簇存索引
    clusters = []
    for i in range(1, labels.max() + 1):
        idxs = np.where(labels == i)[0].tolist()
        clusters.append(idxs)

    return clusters


def cluster_boundary_paths(boundary_paths, distance_threshold=5.0):
    """
    使用 OPTICS 对边界点路径聚类（基于空间距离）

    参数：
        boundary_paths: list of dict，每个 dict 包含 'items': [("l",(x0,y0),(x1,y1)), ...]
        distance_threshold: float，OPTICS 参数 min_samples 的近似距离尺度

    返回：
        clusters: list of lists，每个子列表包含属于同一簇的 path 索引
    """
    n = len(boundary_paths)
    if n == 0:
        return []

    # 提取每条 path 的所有点
    path_points = []
    for path in boundary_paths:
        pts = []
        for item in path['items']:
            if item[0] == 'l':
                pts.append(item[1])
                pts.append(item[2])
        path_points.append(np.array(pts))

    # 计算每条 path 的质心作为代表点
    centroids = np.array([pts.mean(axis=0) for pts in path_points])

    # 使用 OPTICS 聚类
    clustering = OPTICS(min_samples=2, max_eps=distance_threshold, metric='euclidean')
    labels = clustering.fit_predict(centroids)

    # 按 label 分组
    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)

    clusters = list(clusters_dict.values())
    return clusters


def build_building_polygons(boundary_paths, buffer_width=0.2):
    """
    将墙体的边界 paths 转换为 Polygon，返回多边形和每个墙体的通视性值。

    参数:
        boundary_paths: list[dict]，
            每个 dict 包含:
                'items': [("l",(x0,y0),(x1,y1)), ...]
                'hollow': bool 是否为空心墙 (可选，默认 False)
        buffer_width: float，膨胀宽度，用于从线条生成墙体多边形

    返回:
        results: list[dict]，每个元素格式：
            {
                "polygon": Polygon,
                "transparency": float  # 0~1 之间，表示该墙体的通视性
            }
    """

    results = []

    for path in boundary_paths:
        points = []
        hollow_lengths = 0.0
        total_lengths = 0.0

        for item in path.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                points.append(p1)
                points.append(p2)

                # 计算线段长度
                seg_length = LineString([p1, p2]).length
                total_lengths += seg_length

                if path.get("visibility", False):
                    hollow_lengths += seg_length * 0.6

        # 去重连续重复点
        points = [points[i] for i in range(len(points)) if i == 0 or points[i] != points[i - 1]]

        # 少于3点，直接跳过
        if len(points) < 3:
            continue

        try:
            line = LineString(points)
            poly = line.buffer(buffer_width, cap_style=2, join_style=2)

            # 通视性计算
            transparency = hollow_lengths / total_lengths if total_lengths > 0 else 0.0

            if isinstance(poly, MultiPolygon):
                for sub_poly in poly.geoms:
                    results.append({
                        "polygon": sub_poly,
                        "transparency": transparency
                    })
            elif isinstance(poly, Polygon):
                results.append({
                    "polygon": poly,
                    "transparency": transparency
                })

        except Exception as e:
            print("Error building polygon:", e)
            continue

    return results

