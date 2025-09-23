import ezdxf
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from fontTools.ttx import process
from joblib import Parallel, delayed
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from shapely.geometry import Polygon as ShapelyPolygon
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

import pdf_test
from Rs_type import LandscapeCollection, LandscapeElement, Geometry
from matplotlib.patches import Polygon as MplPolygon
import construct

import bound_shape
import clusters
import load
import itertools

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

dxf_files = [
    "赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/2. 留园/2-留园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/3. 寄畅园/2-寄畅园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/4. 瞻园/2-瞻园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/5. 豫园/2-豫园矢量平面图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/6. 秋霞圃/2-秋霞圃平面矢量图(1).dxf",
    # "赛题F江南古典园林美学特征建模附件资料/7. 沈园/2-沈园平面矢量图(1).dxf",
    # "赛题F江南古典园林美学特征建模附件资料/8. 怡园/2-怡园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/9. 耦园/2-耦园平面矢量图.dxf",
    # "赛题F江南古典园林美学特征建模附件资料/10. 绮园/2-绮园平面矢量图.dxf"
]

def plot_rocks_with_transparency(landscape: LandscapeCollection):
    """
    绘制 LandscapeCollection 中的假山元素，并在中心标注 transparent 值
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for element in landscape.get_by_type("rock"):
        geom = element.geometry
        if geom.type != "Polygon":
            continue

        coords = np.array(geom.coordinates)
        # 绘制假山外轮廓
        patch = MplPolygon(coords, closed=True, facecolor="brown", edgecolor="black", alpha=0.5)
        ax.add_patch(patch)

        # 计算簇中心（多边形重心）
        polygon = ShapelyPolygon(coords)
        cx, cy = polygon.centroid.x, polygon.centroid.y

        # 标注 transparent 值
        ax.text(cx, cy, f"{element.transparent:.1f}", color="white",
                fontsize=12, ha="center", va="center", fontweight="bold")

    ax.set_aspect('equal')
    ax.grid(True)
    plt.title("假山及其通视性标注")
    plt.show()

def process_rock(group):
    """
    处理假山元素：
    - 聚类假山边界
    - 绘制点和外围边界
    - 将每簇的最外围点存储为 LandscapeElement

    返回：
        LandscapeCollection 对象，包含聚类后的假山 Polygon
    """
    rock_paths = group.get("假山", [])
    boundary_paths = [p for p in rock_paths if "items" in p]

    # 聚类
    rock_clusters = clusters.cluster_rock(boundary_paths, distance_threshold=1.0)

    base_colors = plt.colormaps["tab20"].colors
    color_cycle = itertools.cycle(base_colors)
    for i, cluster in enumerate(rock_clusters):
        cluster_color = next(color_cycle)

        # 收集簇内所有点
        points = []
        for idx in cluster:
            path = boundary_paths[idx]
            for item in path['items']:
                if item[0] == 'l':
                    points.append(item[1])
                    points.append(item[2])
        points = np.array(points)

        # 计算外围边界
        hull_points = bound_shape.alpha_shape(points, 8.0)
        if len(hull_points) > 1:
            hull_points.append(hull_points[0])  # 闭合
            # 构造 LandscapeElement 并加入 collection
            polygon = Polygon(hull_points)
            # 判断簇内部是否包含建筑
            has_building = any(
                ShapelyPolygon(b_element.geometry.coordinates).within(polygon)
                for b_element in landscape.get_by_type("building")
            )
            element = LandscapeElement(
                id=i,
                type="rock",
                geometry=Geometry(type="Polygon", coordinates=hull_points),
                transparent=1.0 if has_building else 0.0,  # 如果有建筑就透明，否则不透明
                weight=1.5, # TODO 定权待定
                name=f"假山簇 {i + 1}",
                cluster_id=i
            )
            landscape.add_element(element)

    #plot_rocks_with_transparency(landscape)

def cluster_plant(group):
    # 假设 paths 已经是解析 DXF 得到的路径，且只保留“假山”图层
    plant_paths = group.get("植被", [])
    circle_paths = [p for p in plant_paths if "center" in p and "radius" in p]
    # plant_clusters = clusters.cluster_plant_circles(circle_paths, distance_threshold=10.0, overlap_ratio_threshold=0.5)
    # plant_clusters=clusters.hierarchical_cluster_plant(circle_paths, distance_threshold=8.0)
    plant_clusters = clusters.density_cluster_plant(circle_paths,8,2,1)

    n_clusters = len(plant_clusters)
    base_colors = plt.colormaps["tab20"].colors
    # 如果簇数量超过 20，就循环使用
    color_cycle = itertools.cycle(base_colors)

    plt.figure(figsize=(8, 8))
    plt.grid()
    for i, cluster in enumerate(plant_clusters):
        color = next(color_cycle)
        xs, ys = [], []
        for idx in cluster:
            c = circle_paths[idx]["center"]
            xs.append(c[0])
            ys.append(c[1])
        plt.scatter(xs, ys, color=color, s=30, label=f'Cluster {i + 1}') #: {cluster}

    garden_name = load.get_garden_name_from_path(dxf_file)
    plt.title(f"{garden_name} - 植被圆聚类结果")
    plt.axis("equal")
    plt.legend()

def process_plant(group):
    """
    处理植被元素，将所有植被圆直接加入到 LandscapeCollection。
    数据管理方式与假山统一。

    参数:
        group: dict，从 DXF 中解析后的分层数据
        landscape_collection: LandscapeCollection 实例，用于收集所有景观元素
    """
    plant_paths = group.get("植被", [])
    circle_paths = [p for p in plant_paths if "center" in p and "radius" in p]

    for i, p in enumerate(circle_paths):
        # 构造 LandscapeElement
        element = LandscapeElement(
            id=i,
            type="plant",
            geometry=Geometry(type="Circle",center=p["center"], radius=p["radius"]),
            transparent=0.4,  # 植被不可通视
            weight=1.0,  # TODO 可设置权重
            name=f"植被 {i + 1}",
            cluster_id=None  # 暂不聚类
        )
        # 加入 collection
        landscape.add_element(element)

def process_building(gourp):
    building_paths = []
    for key in ["实体墙", "空心墙"]:
        building_paths.extend(gourp.get(key, []))
    boundary_paths = [p for p in building_paths if "items" in p]
    # 构造建筑多边形
    building_res = clusters.build_building_polygons(boundary_paths, 0.2)

    # transparency_list = []
    for i, res in enumerate(building_res):
        poly = res["polygon"]
        transparency = res["transparency"]
        # transparency_list.append(transparency)
        if not isinstance(poly, Polygon):
            continue
        # 获取外轮廓坐标
        x, y = poly.exterior.xy
        coords = list(zip(x, y))[:-1]
        # 构造 LandscapeElement
        element = LandscapeElement(
            id=i,
            type="building",
            geometry=Geometry(type="Polygon", coordinates=coords),
            transparent=transparency,
            weight=2.0,  # TODO: 权重待调整
            name=f"建筑 {i + 1}",
            cluster_id=None
        )
        landscape.add_element(element)

    # # 绘制 transparency 直方图
    # plt.figure(figsize=(8, 5))
    # plt.hist(transparency_list, bins=10, color='skyblue', edgecolor='black')
    # plt.xlabel("Transparency")
    # plt.ylabel("数量")
    # plt.title("建筑元素透明度分布直方图")
    # plt.grid(True)
    # plt.show()

from shapely.geometry import Polygon
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ===== 辅助函数 =====
def path_to_polygon(path) -> Optional[Polygon]:
    """将沿岸线 path 转 Polygon 并闭合"""
    points = []
    for item in path['items']:
        if item[0] == 'l':
            points.append(item[1])
            points.append(item[2])
    # 至少 3 个点才能构成 Polygon
    if len(points) < 3:
        return None
    # 保证闭合
    if points[0] != points[-1]:
        points.append(points[0])
    return Polygon(points)

# ===== 水体处理函数 =====
def process_water(group):
    """
    处理水体图层：
    - 全局聚合所有路径
    - 构造 Polygon
    - 判断全局包含关系生成 holes
    - 构造 LandscapeElement 并加入 collection
    """
    water_paths = group.get("水体", [])
    boundary_paths = [p for p in water_paths if "items" in p]
    if not boundary_paths:
        return

    # 聚类 (假设 clusters.cluster_rock 可用)
    water_clusters = clusters.cluster_rock(boundary_paths, distance_threshold=0.5)

    # ===== 1. 全局生成 polygon 列表 =====
    all_polygons = []
    polygon_to_cluster = {}  # Polygon -> cluster_idx
    for cluster_idx, cluster in enumerate(water_clusters):
        for idx in cluster:
            path = boundary_paths[idx]
            poly = path_to_polygon(path)
            if poly:
                all_polygons.append(poly)
                polygon_to_cluster[id(poly)] = cluster_idx  # 用 id(poly) 作为 key

    if not all_polygons:
        return

    # ===== 2. 全局判断包含关系 =====
    outer_polygons = []
    inner_map = {}  # key: outer polygon idx, value: list of inner polygons

    for i, poly_i in enumerate(all_polygons):
        contained = False
        for j, poly_j in enumerate(all_polygons):
            if i == j:
                continue
            if poly_j.contains(poly_i):
                inner_map.setdefault(j, []).append(poly_i)
                contained = True
                break
        if not contained:
            outer_polygons.append((i, poly_i))

    # ===== 3. 构造 LandscapeElement =====
    for outer_idx, outer_poly in outer_polygons:
        holes = [list(p.exterior.coords) for p in inner_map.get(outer_idx, [])]
        cluster_idx = polygon_to_cluster[id(outer_poly)]
        element = LandscapeElement(
            id=len(landscape.elements),
            type="water",
            geometry=Geometry(
                type="Polygon",
                coordinates={
                    "shell": list(outer_poly.exterior.coords),
                    "holes": holes
                }
            ),
            transparent=1.0,
            weight=1.5,
            name=f"水体簇 {len(landscape.elements)+1}",
            cluster_id=cluster_idx
        )
        landscape.add_element(element)

from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt

def plot_landscape(collection, nodes, edges, figsize=(10, 10)):
    """
    可视化 LandscapeCollection 中的元素
    """
    color_map = {"rock": "red", "building": "black", "plant": "green", "water": "blue"}

    # 创建新图窗
    plt.figure(figsize=figsize)
    plt.gca().set_aspect("equal")
    plt.grid(True)

    for element in collection.elements:
        geom = element.geometry
        color = color_map.get(element.type, "gray")

        if geom.type == "Polygon":
            if element.type == "water":
                shell = list(geom.coordinates["shell"])
                holes = [list(h) for h in geom.coordinates.get("holes", [])]

                # 绘制外环
                patch = MplPolygon(shell, closed=True, alpha=0.4, color=color, label=element.type)
                plt.gca().add_patch(patch)

                # 绘制 holes（只画边界，不填充）
                for hole in holes:
                    hole_patch = MplPolygon(hole, closed=True, facecolor="white", edgecolor=color)
                    plt.gca().add_patch(hole_patch)

            else:
                coords = geom.coordinates
                if isinstance(coords, list) and len(coords) > 2:
                    xs, ys = zip(*coords)
                    plt.fill(xs, ys, alpha=0.4, color=color, label=element.type)
                    plt.plot(xs, ys, color=color, linewidth=1.5)

        elif geom.type == "Circle":
            center = geom.center
            radius = geom.radius
            circle = plt.Circle(center, radius, color=color, alpha=0.4)
            plt.gca().add_patch(circle)
            plt.plot(center[0], center[1], "o", color=color)

        elif geom.type == "Point":
            x, y = geom.coordinates
            plt.plot(x, y, "o", color=color)

    for i, j in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        plt.plot([x1, x2], [y1, y2], color="#DAA520", linewidth=2, label = "Path")

    # 绘制节点编号（确保每个编号只绘制一次）
    drawn_nodes = set()
    for idx, (x, y) in nodes.items():
        if idx not in drawn_nodes:
            plt.text(x, y, str(idx), fontsize=10, color='black', ha='right', va='bottom')
            drawn_nodes.add(idx)

    # 去掉重复图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    garden_name = load.get_garden_name_from_path(dxf_file)
    plt.title(f"{garden_name} - 景观提取结果")
    # plt.show()


from joblib import Parallel, delayed
from tqdm import tqdm


def compute_feature_vectors_for_nodes(landscape, nodes, **kwargs):
    """
    对所有节点计算特征向量（线程 + 批处理）
    避免进程被杀掉，支持大节点量和复杂景观。

    Args:
        landscape: LandscapeCollection 对象
        nodes: dict {idx: (x, y)}
        n_jobs: 并行线程数
        batch_size: 每次并行处理的节点数量
        **kwargs: 传入 construct_sector_feature_vector 的参数
    Returns:
        feature_dict: {idx: feature_vector}
    """
    feature_dict = {}

    for idx, pt in tqdm(nodes.items(), desc="计算节点特征向量"):
        fv = construct.construct_sector_feature_vector(
            landscape=landscape,
            observation_point=pt,
            **kwargs
        )
        feature_dict[idx] = fv

    return feature_dict

def compute_fv(item, landscape, **kwargs):
    idx, pt = item
    fv = construct.construct_sector_feature_vector(
        landscape=landscape,
        observation_point=pt,
        **kwargs
    )
    return idx, fv

def compute_feature_vectors_for_nodes_mp(landscape, nodes, n_jobs=None, **kwargs):
    """
    使用多进程池计算所有节点的特征向量

    Args:
        landscape: LandscapeCollection 对象
        nodes: dict {idx: (x, y)}
        n_jobs: 并行进程数，默认使用 CPU 核心数
        **kwargs: 传入 construct_sector_feature_vector 的参数
    Returns:
        feature_dict: {idx: feature_vector}
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    nodes_items = list(nodes.items())
    feature_dict = {}

    start_time = time.time()
    print(f"使用 {n_jobs} 个进程计算 {len(nodes_items)} 个节点特征向量...")

    with Pool(n_jobs) as pool:
        # 使用 imap_unordered 可以边计算边显示进度
        for idx, fv in tqdm(pool.imap_unordered(compute_fv, nodes_items), total=len(nodes_items), desc="计算节点特征向量"):
            feature_dict[idx] = fv

    end_time = time.time()
    print(f"特征向量计算完成，总耗时: {end_time - start_time:.2f} 秒")

    return feature_dict

def feature_difference(fv1, fv2, alpha=0.5, beta=0.5):
    """
    计算两个特征向量的异景程度
    - fv1, fv2: dict[str, np.ndarray]
    - alpha: 形态差异权重
    - beta: 总量差异权重
    """
    # 拼接成一个长向量
    vec1 = np.concatenate(list(fv1.values()))
    vec2 = np.concatenate(list(fv2.values()))

    # 模式差异
    pattern_diff = np.linalg.norm(vec1 - vec2, ord=1)  # L1 距离

    # 总量差异
    mag1, mag2 = np.sum(vec1), np.sum(vec2)
    mag_diff = abs(mag1 - mag2)

    return alpha * pattern_diff + beta * mag_diff

def compute_edge_scores(nodes, edges, feature_dict):
    """
    计算所有边的异景程度，带进度显示
    """
    edge_scores = {}
    for (i, j) in tqdm(edges, desc="计算边异景程度", total=len(edges)):
        score = feature_difference(feature_dict[i], feature_dict[j])
        edge_scores[(i, j)] = score
    return edge_scores

def plot_graph_with_scores(nodes, edges, edge_scores):
    """
    绘制图，并在边的中点标注异景程度
    """
    plt.figure(figsize=(10, 8))
    for (i, j) in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        plt.plot([x1, x2], [y1, y2], color="gray", linewidth=1)

        # 在边的中点标注数值
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        score = edge_scores[(i, j)]
        plt.text(mx, my, f"{score:.2f}", color="red", fontsize=8, ha="center")

    # 绘制节点
    for idx, (x, y) in nodes.items():
        plt.scatter(x, y, c="blue", s=30)
        plt.text(x, y, str(idx), color="black", fontsize=8, ha="center", va="bottom")

    plt.gca().set_aspect("equal")
    plt.title("路径图及边的异景程度")
    plt.show()

# 示例使用
for dxf_file in dxf_files:
    print(f"\n=== 解析文件: {dxf_file} ===")

    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    grouped, all_entities = load.parse_and_classify_dxf(msp)
    landscape = LandscapeCollection()
    # 水体
    process_water(grouped)

    # 建筑
    process_building(grouped)

    # 假山处理
    process_rock(grouped)

    # 植被
    process_plant(grouped)


    node_, edge_ = load.dxf_path_to_graph("赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平路线.dxf")

    plot_landscape(landscape, node_, edge_)
    # plt.show()

    # 构造指数衰减函数
    decay_fn = construct.exponential_decay(max_distance=50.0, lam=3.0)
    # for i in tqdm(range(30,35,1)):
    #     f_v = construct.construct_sector_feature_vector(landscape, node_[i], n_sectors=16,decay_func=decay_fn)
    #     construct.plot_sector_feature_vector(f_v, 16)
    #     plt.title(f"Scatter{i}")
    # plt.show()

    # 1. 计算所有节点的特征向量
    feature_dict = compute_feature_vectors_for_nodes_mp(
        landscape,
        node_,  # dxf_path_to_graph 返回的 nodes
        n_sectors=16,
        n_jobs=4,
        rays_per_sector=5,
        max_distance=50.0,
        decay_func=decay_fn
    )

    # 2. 计算所有边的异景程度
    edge_scores = compute_edge_scores(node_, edge_, feature_dict)

    # 3. 绘制结果
    plot_graph_with_scores(node_, edge_, edge_scores)
    plt.show()



# plt.figure(figsize=(8,8))
# # 再绘制每条 boundary_path，并用 idx 显示图例
# n = len(boundary_paths)  # 路径数量
# cmap = plt.get_cmap("hsv")  # 使用连续 colormap
# colors = [cmap(i / n) for i in range(n)]  # 生成 n 种颜色
# for idx, path in enumerate(boundary_paths):
#     color_ = colors[idx]
#     xs, ys = [], []
#     for item in path['items']:
#         if item[0] == 'l':
#             xs.extend([item[1][0], item[2][0]])
#             ys.extend([item[1][1], item[2][1]])
#         # 绘制散点，只显示一次图例
#     plt.scatter(xs, ys, s=20, label=f'Path {idx}', color=color_)
#
# plt.legend()
# plt.axis('equal')
# plt.title("假山边界")
# plt.show()