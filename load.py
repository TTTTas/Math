import numpy as np
from collections import defaultdict

import os

import pandas as pd

mm2m_convr = 1000.0;

# 映射表
layer_mapping = {
    'road': '道路',
    '沿岸线': '水体',
    'water': '水体',
    'archi': '空心墙',
    'archi-1': '空心墙',
    'archiw': '实体墙',
    'archi-w': '实体墙',
    'archi 1': '空心墙',
    'archi w': '实体墙',
    'archive-w': '实体墙',
    'stone': '假山',
    'rock': '假山',
    'plant': '植被'
}

# 各类型的通视性规则
visibility_mapping = {
    "水体": True,
    "实体墙": False,
    "假山": False,
    "空心墙": True,
    "植被": False,
}


def parse_and_classify_dxf(msp):
    """
    解析 DXF 实体并分类为：
    - circles: [{'center':(x,y), 'radius':r, 'layer':layer, 'color':rgb}]
    - boundary_paths: [{'items':[("l",(x0,y0),(x1,y1)), ...], 'layer':layer, 'color':rgb}]
    返回：
        grouped_paths: 按图层归类的字典
        all_entities: 所有有效实体集合
    """
    circles = []
    boundary_paths = []

    all_entities = []

    for e in msp:
        layer = e.dxf.layer

        # 圆
        if e.dxftype() == "CIRCLE":
            c = {"center": (e.dxf.center.x / mm2m_convr, e.dxf.center.y / mm2m_convr),
                 "radius": e.dxf.radius / mm2m_convr,
                 "layer": layer,
                 "visibility": False}
            circles.append(c)
            # print(e.dxf.center.x)
            all_entities.append(c)

        # 边界点
        elif e.dxftype() in ["LINE", "LWPOLYLINE", "POLYLINE", "SPLINE"]:
            path = {"items": [], "layer": layer, "visibility": False}
            if e.dxftype() == "LINE":
                path["items"].append(("l", (e.dxf.start.x / mm2m_convr, e.dxf.start.y / mm2m_convr), (e.dxf.end.x / mm2m_convr, e.dxf.end.y / mm2m_convr)))
            elif e.dxftype() in ["LWPOLYLINE"]:
                points = list(e.get_points())
                points_xy = [(p[0] / mm2m_convr, p[1] / mm2m_convr) for p in points]
                for i in range(len(points_xy) - 1):
                    path["items"].append(("l", points_xy[i], points_xy[i+1]))
                # if getattr(e, "closed", False):
                #     path["items"].append(("l", points_xy[-1], points_xy[0]))
            elif e.dxftype() == "POLYLINE":
                points = []
                for v in e.vertices:
                    x, y = v.dxf.location.x / mm2m_convr, v.dxf.location.y / mm2m_convr
                    points.append((x, y))
                # 转换为线段
                for i in range(len(points) - 1):
                    path["items"].append(("l", points[i], points[i + 1]))
                # if getattr(e, "is_closed", False):  # POLYLINE 是否闭合
                #     path["items"].append(("l", points[-1], points[0]))
            elif e.dxftype() == "SPLINE":
                try:
                    cps = np.array(e.control_points)
                    if len(cps) >= 2:
                        for i in range(len(cps) - 1):
                            path["items"].append(("l", (cps[i][0] / mm2m_convr, cps[i][1] / mm2m_convr), (cps[i+1][0] / mm2m_convr, cps[i+1][1] / mm2m_convr)))
                except Exception:
                    continue
            boundary_paths.append(path)
            all_entities.append(path)

    # 按图层归类
    grouped_paths = defaultdict(list)
    for entity in circles + boundary_paths:
        layer_name = entity["layer"]
        label = layer_mapping.get(layer_name, "未知类型")

        # 根据 label 设置通视性
        entity["visibility"] = visibility_mapping.get(label, False)

        grouped_paths[label].append(entity)

    return grouped_paths, all_entities

def get_garden_name_from_path(filepath: str) -> str:
    """
    从文件路径中提取园林名称，例如：
    '赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平面矢量图.dxf'
    返回：'拙政园'
    """
    # 取父目录
    parent_dir = os.path.basename(os.path.dirname(filepath))
    # 处理 '1. 拙政园' -> '拙政园'
    if '.' in parent_dir:
        garden_name = parent_dir.split('.', 1)[1].strip()
    else:
        garden_name = parent_dir
    return garden_name

import ezdxf

def dxf_path_to_graph(dxf_file, target_layer="path", tol=1e-6):
    """
    读取 dxf 文件，把 target_layer 图层的路径转成无向图
    :param dxf_file: dxf 文件路径
    :param target_layer: 目标图层名（默认 "path"）
    :param tol: 节点坐标合并的容差
    :return: (nodes, edges)
             nodes: dict {index: (x, y)}
             edges: list of (i, j)，无向边的节点索引对
    """
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    # 用于去重的点表
    def round_point(p):
        return (round(p[0]/tol)*tol, round(p[1]/tol)*tol)

    point_index = {}
    nodes = {}
    edges = []
    idx = 0

    def get_node_index(pt):
        nonlocal idx
        rpt = round_point(pt)
        if rpt not in point_index:
            point_index[rpt] = idx
            nodes[idx] = rpt
            idx += 1
        return point_index[rpt]

    for e in msp:
        if e.dxf.layer.lower() != target_layer.lower():
            continue

        if e.dxftype() == "LINE":
            p1 = (e.dxf.start.x / mm2m_convr, e.dxf.start.y / mm2m_convr)
            p2 = (e.dxf.end.x / mm2m_convr, e.dxf.end.y / mm2m_convr)
            i, j = get_node_index(p1), get_node_index(p2)
            edges.append((i, j))

        elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            pts = [(v[0] / mm2m_convr, v[1] / mm2m_convr) for v in e.get_points("xy")]
            for k in range(len(pts)-1):
                i, j = get_node_index(pts[k]), get_node_index(pts[k+1])
                edges.append((i, j))
            if e.closed:  # 封闭多段线
                i, j = get_node_index(pts[-1]), get_node_index(pts[0])
                edges.append((i, j))

        elif e.dxftype() == "SPLINE":
            pts = [(p[0] / mm2m_convr, p[1] / mm2m_convr) for p in e.approximate(50)]  # 取 50 个采样点
            for k in range(len(pts)-1):
                i, j = get_node_index(pts[k]), get_node_index(pts[k+1])
                edges.append((i, j))

    return nodes, edges


def load_graph_from_excel(file_path, offset=(0, 0)):
    """
    从 Excel 加载图结构
    - 第1、2列：节点ID，节点坐标 x,y
    - 第3、4列：边起点、终点
    - 第5列：边经过的点坐标列表，格式 "x1,y1;x2,y2;..."
    """
    df = pd.read_excel(file_path, sheet_name=0, header=None)

    nodes = {}
    edges = []
    edge_lengths = {}
    edge_paths = {}  # 新增：边经过的路径点

    # 解析节点部分（前两列）
    for idx, row in df.iterrows():
        if pd.isna(row[0]) or pd.isna(row[1]):
            continue
        node_id = int(row[0])
        x_str, y_str = str(row[1]).split(",")
        x, y = float(x_str) + offset[0], float(y_str) + offset[1]
        nodes[node_id] = (x, y)

    # 解析边部分（第三、四列及第五列）
    for idx, row in df.iterrows():
        if pd.isna(row[2]) or pd.isna(row[3]):
            continue
        start_str, end_str = str(row[2]).split(",")
        start, end = int(start_str), int(end_str)
        edges.append((start, end))

        # 边长度
        if len(row) > 4 and not pd.isna(row[4]):
            # 解析边经过的坐标列表
            coord_strs = str(row[4]).split(";")
            path_points = []
            for s in coord_strs:
                xs, ys = map(float, s.split(","))
                xs += offset[0]
                ys += offset[1]
                path_points.append((xs, ys))
            edge_paths[(start, end)] = path_points

        length = float(row[3])

        edge_lengths[(start, end)] = length

    return nodes, edges, edge_lengths, edge_paths
