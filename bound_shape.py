import math
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def _circumradius(a, b, c):
    """计算三角形外接圆半径"""
    A = math.dist(b, c)
    B = math.dist(a, c)
    C = math.dist(a, b)
    s = (A + B + C) / 2.0
    area = max(0.0, s*(s-A)*(s-B)*(s-C))**0.5
    if area == 0:
        return float('inf')
    return (A * B * C) / (4.0 * area)

def alpha_shape(points, alpha):
    """
    计算点集的 alpha-shape (凹包)
    :param points: list of (x, y) 点坐标
    :param alpha: 控制参数，越小边界越贴合，越大越接近凸包
    :return: list of (x, y) 构成外边界的点（按顺序）
    """
    if len(points) < 4:
        return points

    pts = np.array(points)
    tri = Delaunay(pts)
    triangles = pts[tri.simplices]

    # 保留外接圆半径 <= alpha 的三角形
    kept = []
    for tri_pts in triangles:
        R = _circumradius(tri_pts[0], tri_pts[1], tri_pts[2])
        if R <= alpha:
            kept.append(Polygon(tri_pts))

    if not kept:
        return list(Polygon(points).convex_hull.exterior.coords)

    # 合并三角形
    union = unary_union(kept)

    # 可能得到多个多边形，取最大的一个
    if isinstance(union, MultiPolygon):
        poly = max(union.geoms, key=lambda p: p.area)
    else:
        poly = union

    # 返回外边界坐标（去掉最后重复的首点）
    return list(poly.exterior.coords[:-1])
