from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict

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
