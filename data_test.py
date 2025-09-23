import ezdxf
from collections import defaultdict
import os

def analyze_multiple_dxf_files(file_paths):
    """
    批量解析 DXF 文件，统计每个文件实体基础信息：
    - 实体类型（LINE, LWPOLYLINE, CIRCLE, ARC, SPLINE...）
    - 图层
    - 颜色（true_color / color index）
    - 线型
    - 闭合状态（针对多段线）
    - 点数量（针对多段线 / 多边形）
    """
    all_file_stats = {}  # 保存每个文件的统计

    for dxf_path in file_paths:
        if not os.path.isfile(dxf_path):
            print(f"文件不存在: {dxf_path}")
            continue

        print(f"\n=== 解析文件: {dxf_path} ===")
        try:
            doc = ezdxf.readfile(dxf_path)
        except Exception as ex:
            print(f"无法读取 DXF 文件: {dxf_path}, 错误: {ex}")
            continue

        msp = doc.modelspace()
        file_stats = defaultdict(list)

        for e in msp:
            entity_info = {}
            entity_info['type'] = e.dxftype()
            entity_info['layer'] = e.dxf.layer
            entity_info['color_index'] = e.dxf.color if e.dxf.hasattr("color") else None
            entity_info['true_color'] = e.dxf.true_color if e.dxf.hasattr("true_color") else None
            entity_info['linetype'] = e.dxf.linetype if e.dxf.hasattr("linetype") else None

            # 针对多段线和多边形
            if e.dxftype() == "LWPOLYLINE":
                points = list(e.get_points())  # 返回 (x, y[, start_width, end_width, bulge])
                entity_info['points'] = [(p[0], p[1]) for p in points]  # 只保留 x,y
                entity_info['point_count'] = len(points)
                entity_info['closed'] = e.closed
            elif e.dxftype() == "POLYLINE":
                points = []
                for v in e.vertices:
                    x, y = v.dxf.location.x, v.dxf.location.y
                    points.append((x, y))
                entity_info['points'] = points
                entity_info['point_count'] = len(points)
                flags = e.dxf.flags if e.dxf.hasattr("flags") else 0
                entity_info['closed'] = bool(flags & 1)  # bit0=1表示闭合
            elif e.dxftype() == "LINE":
                entity_info['start'] = tuple(e.dxf.start)
                entity_info['end'] = tuple(e.dxf.end)
            elif e.dxftype() == "CIRCLE":
                entity_info['center'] = tuple(e.dxf.center)
                entity_info['radius'] = e.dxf.radius
            elif e.dxftype() == "ARC":
                entity_info['center'] = tuple(e.dxf.center)
                entity_info['radius'] = e.dxf.radius
                entity_info['start_angle'] = e.dxf.start_angle
                entity_info['end_angle'] = e.dxf.end_angle
            elif e.dxftype() == "SPLINE":
                entity_info['control_points'] = list(e.control_points) if hasattr(e, 'control_points') else []

            file_stats[entity_info['type']].append(entity_info)

        # 打印该文件统计信息
        print(f"=== 文件: {os.path.basename(dxf_path)} DXF 实体类型统计 ===")
        for etype, entities in file_stats.items():
            print(f"类型: {etype}, 数量: {len(entities)}")
            layers = set(e['layer'] for e in entities)
            print(f"  涉及图层: {layers}")
            colors = set(e['true_color'] or e['color_index'] for e in entities)
            print(f"  涉及颜色: {colors}")

        # 保存到总结果字典
        all_file_stats[os.path.basename(dxf_path)] = file_stats

    return all_file_stats


# 示例用法
# dxf_files = [
#     "赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平面矢量图.dxf",
#     "赛题F江南古典园林美学特征建模附件资料/3. 寄畅园/2-寄畅园平面矢量图.dxf",
# ]

dxf_files = [
    "赛题F江南古典园林美学特征建模附件资料/1. 拙政园/2-拙政园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/2. 留园/2-留园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/3. 寄畅园/2-寄畅园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/4. 瞻园/2-瞻园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/5. 豫园/2-豫园矢量平面图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/6. 秋霞圃/2-秋霞圃平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/7. 沈园/2-沈园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/8. 怡园/2-怡园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/9. 耦园/2-耦园平面矢量图.dxf",
    "赛题F江南古典园林美学特征建模附件资料/10. 绮园/2-绮园平面矢量图.dxf"
]

all_entity_stats = analyze_multiple_dxf_files(dxf_files)
