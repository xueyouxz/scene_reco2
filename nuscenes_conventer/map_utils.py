"""
地图处理工具模块
包含NuScenes地图提取器和相关工具函数

Author: Autonomous Driving Expert
Date: 2024
"""
from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union


class NuscMapExtractor:
    """NuScenes地图真值提取器

    Args:
        data_root (str): nuScenes数据集路径
        roi_size (tuple or list): BEV范围 (x_size, y_size)
    """

    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = [
            'boston-seaport',
            'singapore-hollandvillage',
            'singapore-onenorth',
            'singapore-queenstown'
        ]

        # 初始化地图API
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root,
                map_name=loc
            )
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        # 定义局部patch（nuScenes格式）
        self.local_patch = box(
            -roi_size[0] / 2, -roi_size[1] / 2,
            roi_size[0] / 2, roi_size[1] / 2
        )

    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        """合并邻近的人行横道

        Args:
            ped_geoms (list): 多边形列表

        Returns:
            union_ped_geoms (list): 合并后的人行横道
        """

        def get_rec_direction(geom):
            """获取矩形的主方向"""
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:] - rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()
            return rect_v[longest_v_i], v_len[longest_v_i]

        # 构建空间索引树
        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = list(range(len(ped_geoms)))

        for i, pgeom in enumerate(ped_geoms):
            if i not in remain_idx:
                continue

            # 更新剩余索引
            remain_idx.remove(i)
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            # 查询相邻几何体
            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)

                # 如果方向相似（夹角小于8度），则合并
                if 1 - np.abs(cos) < 0.01:
                    final_pgeom[-1] = final_pgeom[-1].union(o)
                    remain_idx.remove(o_idx)

        # 分割合并结果
        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))

        return results

    def get_map_geom(self,
                     location: str,
                     translation: Union[List, NDArray],
                     rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        """提取给定位置和姿态的地图几何元素

        Args:
            location (str): 城市名称
            translation (array): self2global平移，形状 (3,)
            rotation (array): self2global四元数，形状 (4,)

        Returns:
            geometries (Dict): 按类别提取的几何元素
        """

        # 构建patch box（nuScenes格式）
        patch_box = (
            translation[0], translation[1],
            self.roi_size[1], self.roi_size[0]
        )
        rotation = Quaternion(rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # 提取车道分割线
        lane_dividers = self.map_explorer[location]._get_layer_line(
            patch_box, yaw, 'lane_divider'
        )

        road_dividers = self.map_explorer[location]._get_layer_line(
            patch_box, yaw, 'road_divider'
        )

        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # 提取人行横道
        ped_crossings = []
        ped = self.map_explorer[location]._get_layer_polygon(
            patch_box, yaw, 'ped_crossing'
        )

        for p in ped:
            ped_crossings += split_collections(p)

        # 合并分离的人行横道部分
        ped_crossings = self._union_ped(ped_crossings)

        ped_crossing_lines = []
        for p in ped_crossings:
            # 提取外轮廓以获得闭合多段线
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # 提取边界
        # 将道路段和车道的并集作为可行驶区域
        # 不使用nuScenes的drivable_area层，因为其定义可能模糊
        road_segments = self.map_explorer[location]._get_layer_polygon(
            patch_box, yaw, 'road_segment'
        )
        lanes = self.map_explorer[location]._get_layer_polygon(
            patch_box, yaw, 'lane'
        )

        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])

        drivable_areas = split_collections(drivable_areas)

        # 边界定义为可行驶区域的轮廓
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        return dict(
            divider=all_dividers,  # List[LineString]
            ped_crossing=ped_crossing_lines,  # List[LineString]
            boundary=boundaries,  # List[LineString]
            drivable_area=drivable_areas,  # List[Polygon]
        )


# ========================= map_utils/utils.py =========================
"""
地图处理工具函数
"""

from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import ops
import numpy as np
from scipy.spatial import distance
from typing import List, Optional, Tuple
from numpy.typing import NDArray


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    """分割多重几何体为列表并验证有效性

    Args:
        geom (BaseGeometry): 待分割或验证的几何体

    Returns:
        geometries (List): 几何体列表
    """
    assert geom.geom_type in [
        'MultiLineString', 'LineString', 'MultiPolygon',
        'Polygon', 'GeometryCollection'
    ], f"Unsupported geometry type: {geom.geom_type}"

    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom]
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon],
                              roi_size: Tuple) -> List[LineString]:
    """提取可行驶区域轮廓以获取边界列表

    Args:
        drivable_areas (list): 可行驶区域列表
        roi_size (tuple): BEV范围大小

    Returns:
        boundaries (List): 边界列表
    """
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # 略小于ROI以避免边缘上的意外边界
    local_patch = box(
        -max_x + 0.2, -max_y + 0.2,
        max_x - 0.2, max_y - 0.2
    )

    exteriors = []
    interiors = []

    # 收集所有外部和内部边界
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []

    # 处理外部边界
    for ext in exteriors:
        # 注意：确保所有外部边界为顺时针方向
        # 这样每个边界的右手边是可行驶区域，左手边是人行道
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])

        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)

        assert lines.geom_type in ['MultiLineString', 'LineString']
        results.extend(split_collections(lines))

    # 处理内部边界（岛屿）
    for inter in interiors:
        # 注意：确保所有内部边界为逆时针方向
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])

        lines = inter.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)

        assert lines.geom_type in ['MultiLineString', 'LineString']
        results.extend(split_collections(lines))

    return results


def get_ped_crossing_contour(polygon: Polygon,
                             local_patch: box) -> Optional[LineString]:
    """提取人行横道轮廓以获得闭合多段线

    与`get_drivable_area_contour`不同，此函数确保闭合多段线

    Args:
        polygon (Polygon): 待提取的人行横道多边形
        local_patch (box): 局部patch参数

    Returns:
        line (LineString): 闭合线条，如果为空则返回None
    """
    ext = polygon.exterior

    # 确保逆时针方向
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])

    lines = ext.intersection(local_patch)

    if lines.type != 'LineString':
        # 从交集结果中移除点
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)

        # 同一实例但未连接
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))

            # 连接所有线段
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

    if not lines.is_empty:
        return lines

    return None


# ========================= 额外的辅助函数 =========================

def compute_map_statistics(map_geoms: Dict) -> Dict:
    """计算地图几何元素的统计信息

    Args:
        map_geoms: 地图几何元素字典

    Returns:
        stats: 统计信息字典
    """
    stats = {}

    for geom_type, geom_list in map_geoms.items():
        if geom_type == 'drivable_area':
            # 计算可行驶区域统计
            if geom_list:
                areas = [g.area for g in geom_list if isinstance(g, Polygon)]
                stats[geom_type] = {
                    'count': len(geom_list),
                    'total_area': sum(areas),
                    'avg_area': np.mean(areas) if areas else 0,
                    'max_area': max(areas) if areas else 0,
                    'min_area': min(areas) if areas else 0
                }
        else:
            # 计算线条统计
            if geom_list:
                lengths = [g.length for g in geom_list if isinstance(g, LineString)]
                stats[geom_type] = {
                    'count': len(geom_list),
                    'total_length': sum(lengths),
                    'avg_length': np.mean(lengths) if lengths else 0,
                    'max_length': max(lengths) if lengths else 0,
                    'min_length': min(lengths) if lengths else 0
                }

    return stats


def visualize_map_geoms(map_geoms: Dict, save_path: str = None):
    """可视化地图几何元素

    Args:
        map_geoms: 地图几何元素字典
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(12, 12))

    # 颜色映射
    colors = {
        'divider': 'yellow',
        'ped_crossing': 'red',
        'boundary': 'blue',
        'drivable_area': 'lightgray'
    }

    # 绘制可行驶区域
    if 'drivable_area' in map_geoms:
        for poly in map_geoms['drivable_area']:
            if isinstance(poly, Polygon):
                patch = MPLPolygon(
                    np.array(poly.exterior.coords),
                    facecolor=colors['drivable_area'],
                    edgecolor='none',
                    alpha=0.3
                )
                ax.add_patch(patch)

    # 绘制线条元素
    for geom_type in ['divider', 'ped_crossing', 'boundary']:
        if geom_type in map_geoms:
            lines = []
            for line in map_geoms[geom_type]:
                if isinstance(line, LineString):
                    lines.append(np.array(line.coords))

            if lines:
                lc = LineCollection(
                    lines,
                    colors=colors[geom_type],
                    linewidths=2,
                    label=geom_type
                )
                ax.add_collection(lc)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Map Geometry Visualization')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.

    Args:
        geom (BaseGeometry): geoms to be split or validate.

    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon',
                              'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom, ]
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon],
                              roi_size: Tuple) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
        roi_size (tuple): bev range size

    Returns:
        boundaries (List): list of boundaries.
    '''
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # a bit smaller than roi to avoid unexpected boundaries on edges
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)

    exteriors = []
    interiors = []

    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way

        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']

        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']

        results.extend(split_collections(lines))

    return results


def get_ped_crossing_contour(polygon: Polygon,
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params

    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)

        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))

            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)
    if not lines.is_empty:
        return lines

    return None
