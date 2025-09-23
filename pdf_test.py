from shapely.geometry import Point, Polygon, LineString
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import numpy as np
from typing import Callable, Dict

from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import numpy as np
from typing import Dict, Callable

def construct_sector_feature_vector(
        landscape,
        observation_point: tuple,
        n_sectors: int = 8,
        rays_per_sector: int = 5,
        max_distance: float = 50.0,
        decay_func: Callable[[float], float] = None,
        r_safe: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    构造观赏点的空间分布特征向量（n_sectors个扇形区域），支持多种几何元素，并使用空间索引提高效率。

    Args:
        landscape: LandscapeCollection 对象
        observation_point: (x, y) 元组，观测点坐标
        n_sectors: 扇形数量
        rays_per_sector: 每个扇形中射线数量
        max_distance: 射线最大长度
        decay_func: 距离衰减函数，默认使用指数衰减
        r_safe: 空置域半径，射线初始部分不计算特征

    Returns:
        feature_vector: dict，每个景观类型对应长度为 n_sectors 的 np.ndarray
    """
    if decay_func is None:
        decay_func = lambda d: np.exp(-d / max_distance)

    # 构建空间索引，同时记录元素映射
    geom_elements = []
    element_map = []

    for e in landscape.elements:
        geom = None
        if e.geometry.type == "Polygon":
            coords = e.geometry.coordinates
            try:
                if isinstance(coords, dict) and "shell" in coords:
                    shell = coords["shell"]
                    holes = coords.get("holes", [])
                    # 确保每个点是元组
                    shell = [tuple(p) for p in shell]
                    holes = [[tuple(pt) for pt in hole] for hole in holes]
                    geom = Polygon(shell, holes)
                elif isinstance(coords, list):
                    shell = [tuple(p) for p in coords]
                    geom = Polygon(shell)
            except Exception as ex:
                print(f"Polygon creation failed for element {e.id}: {ex}")
        elif e.geometry.type == "Circle" and e.geometry.center and e.geometry.radius:
            try:
                geom = Point(tuple(e.geometry.center)).buffer(e.geometry.radius, resolution=16)
            except Exception as ex:
                print(f"Circle creation failed for element {e.id}: {ex}")
        elif e.geometry.type == "Polyline" and e.geometry.coordinates:
            try:
                line_coords = [tuple(p) for p in e.geometry.coordinates]
                geom = LineString(line_coords)
            except Exception as ex:
                print(f"Polyline creation failed for element {e.id}: {ex}")

        if geom is not None and isinstance(geom, BaseGeometry):
            geom_elements.append(geom)
            element_map.append(e)

    spatial_index = STRtree(geom_elements)

    feature_vector = {etype: np.zeros(n_sectors) for etype in ['building', 'rock', 'plant', 'water']}
    ox, oy = observation_point
    sector_angles = np.linspace(0, 2 * np.pi, n_sectors + 1)

    # 预计算扇形和射线角度
    sector_rays = []
    for i in range(n_sectors):
        start_angle = sector_angles[i]
        end_angle = sector_angles[i + 1]
        angles = np.linspace(start_angle, end_angle, rays_per_sector, endpoint=False)
        sector_rays.append(angles)

    for sector_idx, angles in enumerate(sector_rays):
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            ray_line = LineString([(ox, oy), (ox + dx * max_distance, oy + dy * max_distance)])
            ray_transparency = 1.0

            # 查询可能交互元素
            candidate_geoms = spatial_index.query(ray_line)
            intersections = []

            for geom in candidate_geoms:
                inter = ray_line.intersection(geom)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    intersections.append((inter, geom))
                elif inter.geom_type in ["MultiPoint", "GeometryCollection"]:
                    for pt in inter.geoms:
                        if pt.geom_type == "Point":
                            intersections.append((pt, geom))

            # 按距离从近到远排序
            intersections.sort(key=lambda x: x[0].distance(Point(ox, oy)))

            for pt, geom in intersections:
                dist = pt.distance(Point(ox, oy))
                if dist <= r_safe:
                    continue
                # 获取对应元素
                idx = geom_elements.index(geom)
                element = element_map[idx]

                # 计算权重
                weight = decay_func(dist) * element.weight * ray_transparency
                feature_vector[element.type][sector_idx] += weight

                # 更新射线通视度
                ray_transparency *= element.transparent
                if element.transparent < 1e-5:
                    break  # 射线被阻挡

    return feature_vector
