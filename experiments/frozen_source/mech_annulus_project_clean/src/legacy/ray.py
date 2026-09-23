import numpy as np
from typing import List, Tuple, Optional

# 定义线段：用 [x1, x2, y1, y2] 表示
LineSegment = Tuple[float, float, float, float]

# 定义光线
class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin    # 光线起点 [x, y]
        self.direction = direction  # 光线方向 [dx, dy]（需归一化）

# 定义包围盒（AABB）
class AABB:
    def __init__(self, min_: np.ndarray, max_: np.ndarray):
        self.min = min_  # 左下角 [x_min, y_min]
        self.max = max_  # 右上角 [x_max, y_max]

# 定义 BVH 节点
class BVHNode:
    def __init__(self):
        self.bbox: Optional[AABB] = None  # 当前节点的包围盒
        self.left: Optional[BVHNode] = None  # 左子树
        self.right: Optional[BVHNode] = None  # 右子树
        self.lines: List[LineSegment] = []  # 叶节点存储的线段
def intersect_line_segment(ray: Ray, line: LineSegment) -> Optional[float]:
    """光线与线段求交，返回交点距离 t（若无交点返回 None）"""
    x1, x2, y1, y2 = line
    p0 = np.array([x1, y1])  # 线段起点
    p1 = np.array([x2, y2])  # 线段终点
    d = p1 - p0  # 线段方向向量

    # 计算光线与线段所在直线的交点
    cross_rd = np.cross(ray.direction, d)
    if np.abs(cross_rd) < 1e-6:  # 平行或接近平行
        return None

    t = np.cross(p0 - ray.origin, d) / cross_rd
    u = np.cross(p0 - ray.origin, ray.direction) / cross_rd

    if t >= 0 and 0 <= u <= 1:  # 交点在线段范围内
        return t
    return None


def compute_line_bbox(line: LineSegment) -> AABB:
    """计算线段的包围盒"""
    x1, x2, y1, y2 = line
    min_ = np.array([min(x1, x2), min(y1, y2)])
    max_ = np.array([max(x1, x2), max(y1, y2)])
    return AABB(min_, max_)


def merge_bboxes(bbox1: AABB, bbox2: AABB) -> AABB:
    """合并两个包围盒"""
    min_ = np.minimum(bbox1.min, bbox2.min)
    max_ = np.maximum(bbox1.max, bbox2.max)
    return AABB(min_, max_)


def build_bvh(lines: List[LineSegment], axis: int = 0) -> BVHNode:
    """递归构建 BVH 树"""
    node = BVHNode()

    if len(lines) <= 4:  # 叶节点：直接存储线段
        node.lines = lines
        node.bbox = compute_line_bbox(lines[0])
        for line in lines[1:]:
            node.bbox = merge_bboxes(node.bbox, compute_line_bbox(line))
        return node

    # 按质心坐标排序（沿当前轴）
    lines.sort(key=lambda line: (line[0] + line[1]) / 2 if axis == 0 else (line[2] + line[3]) / 2)

    # 递归构建左右子树
    mid = len(lines) // 2
    node.left = build_bvh(lines[:mid], (axis + 1) % 2)  # 交替选择分割轴（X/Y）
    node.right = build_bvh(lines[mid:], (axis + 1) % 2)
    node.bbox = merge_bboxes(node.left.bbox, node.right.bbox)
    return node
def intersect_aabb(ray: Ray, bbox: AABB) -> bool:
    """检测光线是否击中包围盒"""
    t_min = (bbox.min - ray.origin) / ray.direction
    t_max = (bbox.max - ray.origin) / ray.direction
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_enter = max(t1[0], t1[1])
    t_exit = min(t2[0], t2[1])
    return t_enter <= t_exit and t_exit >= 0

def trace_ray(ray: Ray, bvh_node: BVHNode) -> Optional[float]:
    """递归检测光线与 BVH 的最近交点"""
    if not intersect_aabb(ray, bvh_node.bbox):
        return None

    if bvh_node.lines:  # 叶节点：检测所有线段
        min_t = None
        for line in bvh_node.lines:
            t = intersect_line_segment(ray, line)
            if t is not None and (min_t is None or t < min_t):
                min_t = t
        return min_t
    else:  # 内部节点：递归检测左右子树
        t_left = trace_ray(ray, bvh_node.left)
        t_right = trace_ray(ray, bvh_node.right)
        if t_left is not None and t_right is not None:
            return min(t_left, t_right)
        return t_left or t_right
def render(width: int, height: int, lines: List[LineSegment]) -> np.ndarray:
    """渲染 2D 线段场景"""
    image = np.zeros((height, width, 3), dtype=np.float32)  # RGB 图像
    bvh = build_bvh(lines)  # 构建 BVH

    for y in range(height):
        for x in range(width):
            # 生成从相机出发的光线（假设相机在 (0,0)，朝向屏幕）
            ray = Ray(origin=np.array([x, y]), direction=np.array([0, 1]))
            t = trace_ray(ray, bvh)
            if t is not None:
                image[y, x] = [1.0, 1.0, 1.0]  # 击中线段则设为白色
    return image
# 定义线段（[x1, x2, y1, y2]）
lines = [
    (10, 90, 10, 10),  # 水平线
    (10, 10, 10, 90),  # 垂直线
    (10, 90, 90, 90),  # 另一水平线
    (90, 90, 10, 90),  # 另一垂直线
    (10, 90, 10, 90),  # 对角线
]

# 渲染 100x100 图像
image = render(100, 100, lines)

# 保存结果（需安装 matplotlib）
import matplotlib.pyplot as plt
plt.savefig("line_rendering.png")
plt.imshow(image, origin='lower')
