from concurrent.futures import ThreadPoolExecutor, as_completed

import ezdxf
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from fontTools.ttx import process
from joblib import Parallel, delayed
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from shapely.geometry import Polygon as ShapelyPolygon
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

import gener_path
import pdf_test
from Rs_type import LandscapeCollection, LandscapeElement, Geometry
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
import construct

import bound_shape
import clusters
import load
import itertools

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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
                weight=wights['rock'], # TODO 定权待定
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
            weight=wights['plant'],  # TODO 可设置权重
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
            weight=wights['building'],  # TODO: 权重待调整
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
            weight=wights['water'],
            name=f"水体簇 {len(landscape.elements)+1}",
            cluster_id=cluster_idx
        )
        landscape.add_element(element)

from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt

def plot_landscape(collection, nodes, edges, edge_path = None, edge_scores=None, plant_paths=None, start_ = None, end_ = None, figsize=(10, 10)):
    """
    可视化 LandscapeCollection 中的元素，并绘制路径、道路及边的异景程度
    
    参数:
        collection: LandscapeCollection 对象
        nodes: Dict[node_id, (x,y)]
        edges: List[(i,j)]
        edge_scores: Dict[(i,j), float] 可选，边异景程度
        plant_paths: List[List[(x,y)]] 道路坐标列表，每条道路为点坐标列表
        figsize: 图像尺寸
    """
    color_map = {"rock": "red", "building": "black", "plant": "green", "water": "blue"}

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid(True)

    # --- 绘制景观元素 ---
    for element in collection.elements:
        geom = element.geometry
        color = color_map.get(element.type, "gray")

        if geom.type == "Polygon":
            if element.type == "water":
                shell = list(geom.coordinates["shell"])
                holes = [list(h) for h in geom.coordinates.get("holes", [])]

                patch = MplPolygon(shell, closed=True, alpha=0.4, color=color, label=element.type)
                ax.add_patch(patch)

                for hole in holes:
                    hole_patch = MplPolygon(hole, closed=True, facecolor="white", edgecolor=color)
                    ax.add_patch(hole_patch)
            else:
                coords = geom.coordinates
                if isinstance(coords, list) and len(coords) > 2:
                    xs, ys = zip(*coords)
                    ax.fill(xs, ys, alpha=0.4, color=color, label=element.type)
                    ax.plot(xs, ys, color=color, linewidth=1.5)

        elif geom.type == "Circle":
            center = geom.center
            radius = geom.radius
            circle = plt.Circle(center, radius, color=color, alpha=0.4)
            ax.add_patch(circle)
            ax.plot(center[0], center[1], "o", color=color, label=element.type)

        elif geom.type == "Point":
            x, y = geom.coordinates
            ax.plot(x, y, "o", color=color)

    # --- 绘制道路 ---
    if plant_paths:
        for path in plant_paths:
            for item in path['items']:
                _, start_p, end_p = item
                xs = [start_p[0], end_p[0]]
                ys = [start_p[1], end_p[1]]
                ax.plot(xs, ys, color="#DAA520", linewidth=2, label='Road')

    from scipy.interpolate import splprep, splev

    # --- 绘制边和异景标注 ---
    for i, j in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        path_points = None
        if edge_path:
            if (i, j) in edge_path:
                path_points = edge_path[(i, j)]
            elif (j, i) in edge_path:
                # 顺序反转
                path_points = list(reversed(edge_path[(j, i)]))

        if path_points:
            # 将起点和终点加入路径
            pts = path_points

            xs, ys = zip(*pts)

            if len(xs) >= 4:
                tck, u = splprep([xs, ys], s=0)
                unew = np.linspace(0, 1.0, 100)
                out = splev(unew, tck)
                xs_smooth, ys_smooth = out[0], out[1]
            else:
                xs_smooth, ys_smooth = xs, ys

            ax.plot(xs_smooth, ys_smooth, color="#F0EE80", linewidth=2, label="Path")

            # 起点终点标注不同颜色
            ax.plot(xs_smooth[1], ys_smooth[1], "o", color="#4FA6FE", markersize=6)
            ax.plot(xs_smooth[-2], ys_smooth[-2], "o", color="#DA7878", markersize=6)

        else:
            # 没有路径点，使用原来的直线
            ax.plot([x1, x2], [y1, y2], color="#F0EE80", linewidth=2, label="Path")
            # 箭头头部位置
            arrow_x = x1 + 2/3 * (x2 - x1)
            arrow_y = y1 + 2/3 * (y2 - y1)
            dx, dy = x2 - x1, y2 - y1
            arrow = FancyArrowPatch(
                posA=(arrow_x, arrow_y),
                posB=(arrow_x + dx*1e-2, arrow_y + dy*1e-2),
                arrowstyle='-|>',
                mutation_scale=24,
                color="#797979",
                linewidth=0,
                zorder=10
            )
            ax.add_patch(arrow)

            # 边的异景分数标注
            if edge_scores is not None:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                score = edge_scores.get((i, j)) or edge_scores.get((j, i))
                if score is not None:
                    ax.text(mx, my, f"{score:.2f}", color="red", fontsize=8, ha="center", va="center")


    # --- 绘制节点编号 ---
    drawn_nodes = set()
    for idx, (x, y) in nodes.items():
        if idx not in drawn_nodes:
            ax.text(x, y, str(idx), fontsize=10, color='black', ha='right', va='bottom')
            drawn_nodes.add(idx)

    # --- 绘制起点和终点 ---
    if start_ is not None and end_ is not None:
        if start_ == end_:  # 起点和终点是同一个
            x, y = nodes[start_]
            ax.scatter(x, y, color="purple", s=80, zorder=5, label="Start/End")
        else:
            xs, ys = nodes[start_]
            xe, ye = nodes[end_]
            ax.scatter(xs, ys, color="#476DF8", s=80, zorder=5, label="Start")
            ax.scatter(xe, ye, color="#DE2A2A", s=80, zorder=5, label="End")

    # --- 去掉重复图例 ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title("景观及路径可视化")
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
        fv = construct.construct_sector_feature_vector_fast(
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

    score = alpha * pattern_diff + beta * mag_diff
    if score == 0.0:
        score = 1e-5

    return score

def compute_edge_scores(nodes, edges, feature_dict):
    """
    计算所有边的异景程度，带进度显示
    """
    edge_scores = {}
    for (i, j) in tqdm(edges, desc="计算边异景程度", total=len(edges)):
        score = feature_difference(feature_dict[i], feature_dict[j])
        edge_scores[(i, j)] = score
    return edge_scores

def find_closest_node(nodes_coords, point):
    """
    nodes_coords: dict[node_id, (x,y)]
    point: (x,y)
    """
    coords = np.array(list(nodes_coords.values()))
    node_ids = list(nodes_coords.keys())
    dists = np.linalg.norm(coords - np.array(point), axis=1)
    closest_idx = np.argmin(dists)
    return node_ids[closest_idx]

if __name__ == '__main__':
    wights = {'building': 2.0, 'rock': 1.5, 'plant': 0.65, 'water': 1.5}
    # 每个园林的 DXF 文件、路径 Excel 文件、坐标偏移
    garden_data = [
        # {
        #     "name": "拙政园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/1. 拙政园/keypoints(3).xlsx",
        #     "offset": (9.8491329, -10.8432441),
        #     "start": (151.2011329,9.7517559),
        #     "end": (151.2011329,9.7517559)
        # },
        # {
        #     "name": "留园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/2. 留园/2-留园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/2. 留园/keypoints_15_6.xlsx",
        #     "offset": (-0.9796737, -0.0270097),
        #     "start": (91.98,15.625),
        #     "end": (91.98,15.625)
        # },
        # {
        #     "name": "寄畅园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/3. 寄畅园/2-寄畅园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/3. 寄畅园/keypoints_8_5.xlsx",
        #     "offset": (-0.1977373, -0.2680043),
        #     "start": (60.02,103.43),
        #     "end": (60.02,103.43)
        # },
        # {
        #     "name": "瞻园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/4. 瞻园/2-瞻园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/4. 瞻园/keypoints_8_5.xlsx",
        #     "offset": (-1.9268614, -0.8434156),
        #     "start": (92.7846474,0.5689201),
        #     "end": (92.7846474,0.5689201)
        # },
        # {
        #     "name": "豫园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/5. 豫园/2-豫园矢量平面图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/5. 豫园/keypoints_12_5.xlsx",
        #     "offset": (-3.2157811, -21.6134844),
        #     "start": (-11.2,57.45),
        #     "end": (-75.8,129.7)
        # },
        # {
        #     "name": "秋霞圃",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/6. 秋霞圃/2-秋霞圃平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/6. 秋霞圃/keypoints_8_5.xlsx",
        #     "offset": (-7.6288985, -7.1803547),
        #     "start": (101.2160972,13.208431),
        #     "end": (101.2160972,13.208431)
        # },
        # {
        #     "name": "沈园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/7. 沈园/2-沈园平面矢量图(1).dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/7. 沈园/keypoints_10_5.xlsx",
        #     "offset": (-0.9630922, -4.5601914),
        #     "start": (60.12,135.74),
        #     "end": (60.12,135.74)
        # },
        {
            "name": "怡园",
            "dxf_file": "赛题F江南古典园林美学特征建模附件资料/8. 怡园/2-怡园平面矢量图.dxf",
            "path_file": "赛题F江南古典园林美学特征建模附件资料/8. 怡园/keypoints_8_3.xlsx",
            "offset": (-217.8108044, 5.0781872),
            "start": (60.12,135.74),
            "end": (-148.6982585,1.6179871)
        },
        # {
        #     "name": "耦园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/9. 耦园/2-耦园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/9. 耦园/keypoints_3_4.xlsx",
        #     "offset": (70.3353057, 18.964422),
        #     "start": (61,65.7),
        #     "end": (61,65.7)
        # },
        # {
        #     "name": "绮园",
        #     "dxf_file": "赛题F江南古典园林美学特征建模附件资料/10. 绮园/2-绮园平面矢量图.dxf",
        #     "path_file": "赛题F江南古典园林美学特征建模附件资料/10. 绮园/keypoints_5_8.xlsx",
        #     "offset": (-0.3160398, -4.4043545),
        #     "start": (4.5,6.7),
        #     "end": (4.5,6.7)
        # },
    ]

    for garden in garden_data:
        dxf_file = garden["dxf_file"]
        path_file = garden["path_file"]
        offset = garden["offset"]
        start_point = garden["start"]
        end_point = garden["end"]


        print(f"\n=== 解析园林: {garden['name']} ===")

        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()

        grouped, all_entities = load.parse_and_classify_dxf(msp)
        node_, edge_, edge_length_, edge_paths_ = load.load_graph_from_excel(path_file, offset)
        start_id = find_closest_node(node_, start_point)
        end_id = find_closest_node(node_, end_point)
        
        landscape = LandscapeCollection()
        # 水体
        process_water(grouped)

        # 建筑
        process_building(grouped)

        # 假山处理
        process_rock(grouped)

        # 植被
        process_plant(grouped)

        # print(next(iter(edge_paths_.values())))
        # input()
        # plot_landscape(landscape, node_, edge_)

        road_path = grouped.get("道路", []) 
        plot_landscape(landscape, node_, edge_, edge_path=edge_paths_, start_=start_id, end_=end_id)
        plt.show()

        # 构造指数衰减函数
        decay_fn = construct.exponential_decay(max_distance=50.0, lam=3.0)
        # for i in tqdm(range(60,75,1)):
        #     f_v = construct.construct_sector_feature_vector_fast(landscape, node_[i], n_sectors=16,decay_func=decay_fn)
        #     construct.plot_sector_feature_vector(f_v, 16)
        #     plt.title(f"Scatter{i}")
        # plt.show()
        # f_v = construct.construct_sector_feature_vector_fast(landscape, node_[55], n_sectors=16,decay_func=decay_fn)
        # construct.plot_sector_feature_vector(f_v, 16)
        # plt.title(f"Scatter{55}")
        # plt.show()

        # 1. 计算所有节点的特征向量
        feature_dict = compute_feature_vectors_for_nodes(
            landscape,
            node_,  # dxf_path_to_graph 返回的 nodes
            n_sectors=16,
            rays_per_sector=4,
            max_distance=30.0,
            decay_func=decay_fn
        )

        # 2. 计算所有边的异景程度
        edge_scores = compute_edge_scores(node_, edge_, feature_dict)


        # plot_landscape(landscape, node_, edge_, edge_scores=edge_scores)
        # plt.show()

        # 1. 调用遗传算法规划路径
        best_path, best_score = gener_path.genetic_path_planning(
            node_, edge_, edge_scores, edge_length_,
            start=start_id, end=end_id,feature_dict=feature_dict,
            population_size=100, generations=100,
            alpha=4.2, beta=1.8, gamma=6, delta=0.02, cluster_eps=5.0
        )

        print("最优路径：", best_path)
        print("最优路径综合评分：", best_score)

        # 2. 将路径转为边列表用于绘图
        path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]

        # 提取路径节点（仅保留在路径上的节点）
        path_nodes = {idx: node_[idx] for idx in best_path}
        # 转换为 DataFrame
        df = pd.DataFrame([
            {"node_id": nid, "x": coord[0], "y": coord[1]} 
            for nid, coord in path_nodes.items()
        ])

        # 保存到 Excel
        excel_path = "path_nodes.xlsx"
        df.to_excel(excel_path, index=False)

        print(f"路径节点已保存到 {excel_path}")
        

        # 3. 可视化景观与路径
        plot_landscape(
            collection=landscape,
            nodes=path_nodes,
            edges=path_edges,  # 只绘制路径边
            edge_path=edge_paths_,
            plant_paths=road_path,
            start_=start_id, end_=end_id,
            figsize=(12, 12)
        )
        plt.show()
