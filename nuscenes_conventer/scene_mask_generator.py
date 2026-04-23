import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops as ops
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm

from config_loader import load_config
from nuscenes_conventer.map_utils import NuscMapExtractor


def _smooth_trajectory(positions: np.ndarray) -> np.ndarray:
    """平滑轨迹"""
    # 使用简单的移动平均进行平滑
    window_size = 3
    if len(positions) < window_size:
        return positions

    smoothed = np.zeros_like(positions)
    for i in range(len(positions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)

    return smoothed


class SceneMaskGenerator:
    """场景级地图掩码生成器
    
    主要功能：
    1. 提取场景中的地图图层(drivable_area, ped_crossing, divider, boundary)
    2. 提取场景中自车的完整轨迹
    3. 将地图元素和轨迹栅格化为掩码
    4. 保存掩码数据并生成可视化
    """

    def __init__(self, config_path: str):
        """初始化掩码生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 从配置中读取参数
        self.grid_resolution = self.config['processing_config']['grid_resolution']
        self.map_size = self.config['processing_config']['map_size']
        self.max_scenes = self.config['processing_config']['max_scenes']

        # 计算栅格尺寸
        self.grid_width = int(self.map_size[0] / self.grid_resolution)
        self.grid_height = int(self.map_size[1] / self.grid_resolution)

        # 获取图层配置
        self.enabled_layers = self.config['map_layers_config']['enabled_layers']

        # 自车配置
        self.ego_size = self.config['ego_vehicle_config']['size']
        self.trajectory_smoothing = self.config['ego_vehicle_config']['trajectory_smoothing']

        # 初始化nuScenes和地图提取器
        self._init_nuscenes()

        # 初始化地图探索器
        self._init_map_explorers()

    def _init_nuscenes(self):
        """初始化nuScenes数据集和地图提取器"""
        version = self.config['dataset_config']['version']
        if not version.startswith('v1.0-'):
            version = f'v1.0-{version}'

        root_path = self.config['dataset_config']['root_path']

        self.nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

        # 初始化地图提取器
        roi_size = tuple(self.map_size)
        self.map_extractor = NuscMapExtractor(root_path, roi_size)

    def _init_map_explorers(self):
        """初始化地图探索器"""
        self.MAPS = [
            'boston-seaport',
            'singapore-hollandvillage',
            'singapore-onenorth',
            'singapore-queenstown'
        ]

        root_path = self.config['dataset_config']['root_path']

        # 初始化地图API和探索器
        self.nusc_maps = {}
        self.map_explorers = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
            self.map_explorers[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

    def generate_scene_masks(self) -> Dict:
        """生成所有场景的掩码数据
        
        Returns:
            scene_masks_data: 包含所有场景掩码数据的字典
        """

        # 获取要处理的场景
        scenes_to_process = self._get_scenes_to_process()

        scene_masks_data = {
            'scenes': {},
            'metadata': self._build_metadata(),
            'config': self.config
        }

        # 创建输出目录
        output_dir = Path(self.config['output_config']['save_path'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理每个场景
        for scene_idx, scene in enumerate(tqdm(scenes_to_process, desc="Processing scenes")):
            if self.max_scenes > 0 and scene_idx >= self.max_scenes:
                break

            scene_token = scene['token']
            scene_name = scene['name']
            scene_mask_data = self._generate_single_scene_mask(scene)
            scene_masks_data['scenes'][scene_token] = scene_mask_data

            # 如果启用可视化，生成可视化图像
            if self.config['visualization_config']['generate_visualizations']:
                self._visualize_scene_mask(
                    scene_mask_data,
                    output_dir / f"visualizations/{scene_name}_mask.png"
                )
        return scene_masks_data

    def _get_scenes_to_process(self) -> List[Dict]:
        """获取要处理的场景列表"""
        data_split = self.config['dataset_config']['data_split']
        version = self.config['dataset_config']['version']

        # 获取场景分割
        from nuscenes.utils import splits

        if version == 'trainval':
            if data_split == 'train':
                scene_names = splits.train
            elif data_split == 'val':
                scene_names = splits.val
            else:
                scene_names = splits.train + splits.val
        elif version == 'test':
            scene_names = splits.test
        elif version == 'mini':
            if data_split == 'train':
                scene_names = splits.mini_train
            elif data_split == 'val':
                scene_names = splits.mini_val
            else:
                scene_names = splits.mini_train + splits.mini_val
        else:
            raise ValueError(f'Unknown version: {version}')

        # 过滤可用场景
        available_scenes = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                available_scenes.append(scene)

        return available_scenes

    def _smooth_polygon_boundary(self, polygon: Polygon) -> Polygon:
        """
        使用buffer方法平滑多边形边界并填充凹陷区域
        """
        if not polygon.is_valid or polygon.is_empty:
            return polygon

        try:
            # 第一阶段：填充小的凹陷和间隙
            small_buffer_distance = 2.0  # 2米的buffer用于填充小凹陷
            stage1_polygon = polygon.buffer(small_buffer_distance).buffer(-small_buffer_distance)

            # 第二阶段：进一步平滑边界
            medium_buffer_distance = 5.0  # 5米的buffer用于显著平滑
            stage2_polygon = stage1_polygon.buffer(medium_buffer_distance).buffer(-medium_buffer_distance)

            # 第三阶段：处理较大的凹陷区域
            large_buffer_distance = 8.0  # 8米的buffer用于填充较大凹陷
            stage3_polygon = stage2_polygon.buffer(large_buffer_distance).buffer(-large_buffer_distance)

            # 验证处理结果
            if stage3_polygon.is_valid and not stage3_polygon.is_empty:
                final_polygon = stage3_polygon
            elif stage2_polygon.is_valid and not stage2_polygon.is_empty:
                final_polygon = stage2_polygon
            elif stage1_polygon.is_valid and not stage1_polygon.is_empty:
                final_polygon = stage1_polygon
            else:
                final_polygon = polygon

            # 如果结果是MultiPolygon，选择最大的部分
            if final_polygon.geom_type == 'MultiPolygon':
                largest_polygon = max(final_polygon.geoms, key=lambda p: p.area)
                final_polygon = largest_polygon

            # 最后的修复检查
            if not final_polygon.is_valid:
                final_polygon = final_polygon.buffer(0)

            return final_polygon

        except Exception as e:
            print(f"警告：多边形平滑处理时出现异常: {e}")
            return polygon

    def get_patch_coord(self, patch_box: tuple, patch_angle: float = 0.0) -> Polygon:
        """
        将patch box转换为Polygon对象
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        # 创建矩形
        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)

        # 如果有角度，进行旋转
        if patch_angle != 0.0:
            patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y))

        return patch

    def get_sample_patch_box(self, ego_poses: List[Dict]) -> List[Polygon]:
        """
        根据自车位姿计算每个样本的patch box
        """
        sample_patch_boxes = []
        for ego_pose in ego_poses:
            translation = ego_pose['translation']
            rotation = Quaternion(ego_pose['rotation'])
            yaw = quaternion_yaw(rotation) / np.pi * 180
            patch_box = (translation[0], translation[1], self.map_size[1], self.map_size[0])
            sample_patch_boxes.append(self.get_patch_coord(patch_box, yaw))
        return sample_patch_boxes

    def get_map_range(self, patch_boxes: List[Polygon]) -> Polygon:
        """
        将多个patch做合并，得到一个全局的map_range，并进行边界平滑处理
        """
        # 过滤掉无效的多边形
        valid_patches = [patch for patch in patch_boxes if patch.is_valid and not patch.is_empty]

        try:
            # 使用shapely的unary_union合并所有patch box
            unified_polygon = ops.unary_union(valid_patches)

            # 如果结果是MultiPolygon，取最大的部分
            if unified_polygon.geom_type == 'MultiPolygon':
                largest_polygon = max(unified_polygon.geoms, key=lambda p: p.area)
                unified_polygon = largest_polygon

            # 检查合并后的多边形是否有效
            if not unified_polygon.is_valid:
                unified_polygon = unified_polygon.buffer(0)

            # 平滑边界和填充凹陷区域
            smoothed_polygon = self._smooth_polygon_boundary(unified_polygon)

            # 添加适当的边界扩展
            patch_margin = 10.0  # 10米的边界扩展
            expanded_polygon = smoothed_polygon.buffer(patch_margin)

            # 确保最终结果是有效的多边形
            if expanded_polygon.is_valid and not expanded_polygon.is_empty:
                return expanded_polygon
            else:
                return smoothed_polygon if smoothed_polygon.is_valid else unified_polygon

        except Exception as e:
            print(f"警告：合并patch box时出现异常: {e}")
            # 返回所有patch的边界框
            all_coords = []
            for patch in valid_patches:
                all_coords.extend(list(patch.exterior.coords))
            if all_coords:
                coords_array = np.array(all_coords)
                min_x, min_y = coords_array.min(axis=0)
                max_x, max_y = coords_array.max(axis=0)
                return box(min_x, min_y, max_x, max_y)
            else:
                center_x, center_y = 0, 0
                return box(center_x - self.map_size[0] / 2, center_y - self.map_size[1] / 2,
                           center_x + self.map_size[0] / 2, center_y + self.map_size[1] / 2)

    def _generate_single_scene_mask(self, scene: Dict) -> Dict:
        """生成单个场景的掩码数据
        
        Args:
            scene: 场景信息字典
            
        Returns:
            scene_mask_data: 场景掩码数据
        """
        scene_token = scene['token']
        scene_name = scene['name']

        # 获取场景中的所有样本
        samples = self._get_scene_samples(scene_token)

        # 提取自车轨迹
        ego_trajectory = self._extract_ego_trajectory(samples)

        # 获取场景的地图位置
        map_location = self.nusc.get('log', scene['log_token'])['location']

        # 基于自车轨迹计算地图范围
        ego_poses = []
        for sample in samples:
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = self.nusc.get('sample_data', lidar_token)
            pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
            ego_poses.append(pose_record)

        # 计算patch boxes和地图范围
        sample_patch_boxes = self.get_sample_patch_box(ego_poses)
        map_range = self.get_map_range(sample_patch_boxes)

        # 使用nuScenes-devkit的方式生成多通道地图掩码
        map_masks = self._generate_map_mask(map_location, map_range)

        # 生成自车轨迹掩码
        ego_mask = self._rasterize_ego_trajectory(ego_trajectory, map_range)

        # 添加自车轨迹掩码到多通道掩码字典
        map_masks['ego_trajectory'] = ego_mask


        # 计算地图范围的边界框
        bounds = map_range.bounds  # (minx, miny, maxx, maxy)
        range_center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        range_size = [bounds[2] - bounds[0], bounds[3] - bounds[1]]

        return {
            'scene_token': scene_token,
            'scene_name': scene_name,
            'map_location': map_location,
            'map_range': map_range,
            'range_center': range_center,
            'range_size': range_size,
            'ego_trajectory': ego_trajectory,
            'masks': map_masks,  # 多通道掩码字典
            'grid_info': {
                'resolution': self.grid_resolution,
                'width': self.grid_width,
                'height': self.grid_height,
                'origin': [bounds[0], bounds[1]]
            }
        }

    def _get_scene_samples(self, scene_token: str) -> List[Dict]:
        """获取场景中的所有样本"""
        scene = self.nusc.get('scene', scene_token)

        samples = []
        sample_token = scene['first_sample_token']

        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']

        return samples

    def _extract_ego_trajectory(self, samples: List[Dict]) -> Dict:
        """提取自车轨迹"""
        positions = []
        rotations = []
        timestamps = []

        for sample in samples:
            # 获取lidar数据
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = self.nusc.get('sample_data', lidar_token)
            pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

            positions.append(pose_record['translation'])
            rotations.append(pose_record['rotation'])
            timestamps.append(sample['timestamp'])

        positions = np.array(positions)
        rotations = np.array(rotations)
        timestamps = np.array(timestamps)

        # 如果启用轨迹平滑
        if self.trajectory_smoothing:
            positions = _smooth_trajectory(positions)

        return {
            'positions': positions,
            'rotations': rotations,
            'timestamps': timestamps
        }

    def _generate_map_mask(self, map_location: str, map_range: Polygon) -> Dict[str, np.ndarray]:
        """
        使用多边形范围生成地图掩码，并对地图元素进行裁剪
        
        Returns:
            Dict[str, np.ndarray]: 每个图层一个掩码的字典
        """
        # 获取地图范围的边界框
        bounds = map_range.bounds  # (minx, miny, maxx, maxy)
        range_width = bounds[2] - bounds[0]
        range_height = bounds[3] - bounds[1]


        # 计算画布尺寸
        canvas_width = int(range_width / self.grid_resolution)
        canvas_height = int(range_height / self.grid_resolution)

        # 更新实际的栅格尺寸
        self.grid_width = canvas_width
        self.grid_height = canvas_height

        # 初始化多通道掩码字典
        layer_masks = {}
        for layer in self.enabled_layers:
            layer_masks[layer] = np.zeros((canvas_height, canvas_width), dtype=np.uint8)


        # 获取地图API
        map_api = self.nusc_maps[map_location]

        try:
            # 定义要提取的图层和对应的记录类型
            layer_records = {
                'drivable_area': 'drivable_area',
                'ped_crossing': 'ped_crossing',
                'divider': ['lane_divider', 'road_divider'],
                'boundary': ['road_segment', 'lane']
            }

            # 处理每个启用的图层
            for target_layer in self.enabled_layers:
                if target_layer not in layer_records:
                    continue

                record_names = layer_records[target_layer]
                if isinstance(record_names, str):
                    record_names = [record_names]

                # 提取并合并每个记录类型的几何体
                geometries = []
                for record_name in record_names:
                    try:
                        # 获取在地图范围内的记录
                        records = map_api.get_records_in_patch(
                            map_range.bounds,
                            layer_names=[record_name],
                            mode='intersect'
                        )

                        # 提取几何体并与地图范围做交集
                        for record_token in records[record_name]:
                            try:
                                record = map_api.get(record_name, record_token)

                                # 获取几何体
                                if record_name == 'drivable_area':
                                    # drivable_area有多个多边形，使用polygon_tokens
                                    polygons = []
                                    for polygon_token in record['polygon_tokens']:
                                        poly = map_api.extract_polygon(polygon_token)
                                        if poly.is_valid and not poly.is_empty:
                                            polygons.append(poly)
                                    if polygons:
                                        # 合并多个多边形
                                        geom = unary_union(polygons)
                                    else:
                                        continue
                                elif record_name in ['ped_crossing', 'road_segment', 'lane']:
                                    # 其他多边形图层使用polygon_token
                                    geom = map_api.extract_polygon(record['polygon_token'])
                                elif record_name in ['lane_divider', 'road_divider']:
                                    # 线条图层使用line_token
                                    geom = map_api.extract_line(record['line_token'])
                                else:
                                    continue

                                # 与地图范围做交集裁剪
                                if geom.is_valid and not geom.is_empty:
                                    clipped_geom = geom.intersection(map_range)
                                    if not clipped_geom.is_empty:
                                        geometries.append(clipped_geom)
                            except Exception as e:
                                print(f"警告：处理 {record_name} 记录 {record_token} 时出错: {e}")
                                continue
                    except Exception as e:
                        print(f"警告：处理图层 {record_name} 时出错: {e}")
                        continue

                # 栅格化几何体
                if geometries:
                    try:
                        self._rasterize_geometries(
                            layer_masks[target_layer],
                            geometries,
                            bounds,
                            target_layer
                        )

                        # 更新合并掩码
                    except Exception as e:
                        print(f"警告：栅格化图层 {target_layer} 时出错: {e}")

            return layer_masks
        except Exception as e:
            print(f"警告：生成地图掩码失败: {e}")
            # 回退到空掩码
            return layer_masks

    def _rasterize_geometries(self, mask: np.ndarray, geometries: List, bounds: Tuple, layer_type: str):
        """
        栅格化几何体到掩码中
        
        Args:
            mask: 目标掩码数组
            geometries: 几何体列表
            bounds: 边界框 (minx, miny, maxx, maxy)
            layer_type: 图层类型
        """
        origin_x, origin_y = bounds[0], bounds[1]
        range_width = bounds[2] - bounds[0]
        range_height = bounds[3] - bounds[1]

        for geom in geometries:
            try:
                if geom.is_empty:
                    continue

                # 处理不同类型的几何体
                if geom.geom_type == 'Polygon':
                    self._rasterize_polygon_unified(mask, geom, bounds)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        self._rasterize_polygon_unified(mask, poly, bounds)
                elif geom.geom_type == 'LineString':
                    width = 2
                    self._rasterize_line_unified(mask, geom, bounds, width)
                elif geom.geom_type == 'MultiLineString':
                    width = 2
                    for line in geom.geoms:
                        self._rasterize_line_unified(mask, line, bounds, width)

            except Exception as e:
                print(f"警告：栅格化几何体时出错: {e}")
                continue

    def _rasterize_polygon_unified(self, mask: np.ndarray, polygon: Polygon, bounds: Tuple):
        """使用统一的坐标转换方式栅格化多边形"""
        if polygon.is_empty or not polygon.is_valid:
            return

        # 获取多边形外轮廓坐标
        coords = np.array(polygon.exterior.coords)

        # 转换为像素坐标
        pixel_coords = self._world_to_pixel_unified(coords, bounds)

        # 使用OpenCV填充多边形
        cv2.fillPoly(mask, [pixel_coords.astype(np.int32)], 1)

    def _rasterize_line_unified(self, mask: np.ndarray, line: LineString, bounds: Tuple, width: int):
        """使用统一的坐标转换方式栅格化线条"""
        if line.is_empty:
            return

        # 获取线条坐标
        coords = np.array(line.coords)

        # 转换为像素坐标
        pixel_coords = self._world_to_pixel_unified(coords, bounds)

        # 绘制线条
        for i in range(len(pixel_coords) - 1):
            pt1 = tuple(pixel_coords[i].astype(np.int32))
            pt2 = tuple(pixel_coords[i + 1].astype(np.int32))
            cv2.line(mask, pt1, pt2, 1, width)

    def _world_to_pixel_unified(self, world_coords: np.ndarray, bounds: Tuple) -> np.ndarray:
        """
        统一的世界坐标到像素坐标转换方法
        与地图掩码使用相同的坐标转换方式
        
        Args:
            world_coords: 世界坐标数组 [N, 2]
            bounds: 边界框 (minx, miny, maxx, maxy)
            
        Returns:
            像素坐标数组 [N, 2]
        """
        origin_x, origin_y = bounds[0], bounds[1]
        range_width = bounds[2] - bounds[0]
        range_height = bounds[3] - bounds[1]

        # 转换为相对于边界框的坐标
        relative_x = world_coords[:, 0] - origin_x
        relative_y = world_coords[:, 1] - origin_y

        # 转换为像素坐标
        pixel_x = relative_x / self.grid_resolution
        pixel_y = (range_height - relative_y) / self.grid_resolution  # Y轴翻转

        return np.column_stack([pixel_x, pixel_y])

    def _rasterize_ego_trajectory(self, ego_trajectory: Dict, map_range: Polygon) -> np.ndarray:
        """
        栅格化自车轨迹，使用与地图掩码相同的坐标转换方式
        """
        # 获取地图范围的边界框
        bounds = map_range.bounds  # (minx, miny, maxx, maxy)
        range_width = bounds[2] - bounds[0]
        range_height = bounds[3] - bounds[1]

        # 计算画布尺寸 - 与地图掩码保持一致
        canvas_width = int(range_width / self.grid_resolution)
        canvas_height = int(range_height / self.grid_resolution)

        # 创建掩码
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        positions = ego_trajectory['positions']

        # 获取自车宽度（像素单位）
        ego_width = self.ego_size[1]  # 自车宽度（米）
        line_width = max(1, int(ego_width / self.grid_resolution))  # 转换为像素宽度

        # 将轨迹点转换为像素坐标
        trajectory_coords = np.array(positions)[:, :2]  # 只取x, y坐标
        pixel_coords = self._world_to_pixel_unified(trajectory_coords, bounds)

        # 栅格化轨迹线（宽度与自车宽度相同）
        for i in range(len(pixel_coords) - 1):
            pt1 = tuple(pixel_coords[i].astype(np.int32))
            pt2 = tuple(pixel_coords[i + 1].astype(np.int32))

            # 检查坐标是否在画布范围内
            if (0 <= pt1[0] < canvas_width and 0 <= pt1[1] < canvas_height and
                    0 <= pt2[0] < canvas_width and 0 <= pt2[1] < canvas_height):
                # 使用粗线条绘制轨迹（宽度与自车宽度相同）
                cv2.line(mask, pt1, pt2, 1, line_width)

        return mask

    def _get_ego_corners(self, position: np.ndarray, rotation: np.ndarray,
                         length: float, width: float) -> np.ndarray:
        """获取自车四个角点坐标"""
        # 创建自车边界框的四个角点（相对于车辆中心）
        corners = np.array([
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2]
        ])

        # 应用旋转
        quat = Quaternion(rotation)
        yaw = quaternion_yaw(quat)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])

        # 旋转角点
        rotated_corners = corners @ rotation_matrix.T

        # 平移到全局坐标
        global_corners = rotated_corners + position[:2]

        return global_corners

    def _visualize_scene_mask(self, scene_mask_data: Dict, save_path: Path):
        """可视化场景掩码，仅显示掩码数据"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取配置
        fig_size = self.config['visualization_config']['figure_size']
        dpi = self.config['visualization_config']['dpi']
        color_map = self.config['visualization_config']['color_map']

        # 获取地图范围
        map_range = scene_mask_data['map_range']
        bounds = map_range.bounds  # (minx, miny, maxx, maxy)

        # 获取掩码数据
        masks = scene_mask_data['masks']

        # 创建多通道掩码可视化
        # 确定要显示的图层
        display_layers = []
        for layer in self.enabled_layers:
            if layer in masks:
                display_layers.append(layer)

        # 添加自车轨迹和合并掩码
        if 'ego_trajectory' in masks:
            display_layers.append('ego_trajectory')

        # 计算子图网格大小
        n_layers = len(display_layers)
        n_cols = min(3, n_layers)  # 最多3列
        n_rows = (n_layers + n_cols - 1) // n_cols  # 向上取整

        # 创建图像
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # 确保axes是列表

        # 绘制每个掩码图层
        for i, layer_name in enumerate(display_layers):
            if i < len(axes):  # 确保不超出axes数量
                mask = masks[layer_name]
                # 单通道掩码使用单一颜色
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                if layer_name in color_map:
                    color = color_map[layer_name]
                else:
                    color = [255, 255, 255]  # 默认白色

                colored_mask[mask > 0] = color
                title = f'{layer_name.replace("_", " ").title()} Mask'

                # 显示掩码
                axes[i].imshow(colored_mask, origin='lower', extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
                axes[i].set_title(title)
                axes[i].set_xticks([])  # 移除刻度
                axes[i].set_yticks([])  # 移除刻度

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    def _build_metadata(self) -> Dict:
        """构建元数据"""
        return {
            'version': self.config['dataset_config']['version'],
            'data_split': self.config['dataset_config']['data_split'],
            'grid_resolution': self.grid_resolution,
            'map_size': self.map_size,
            'grid_size': [self.grid_width, self.grid_height],
            'enabled_layers': self.enabled_layers,
            'ego_size': self.ego_size,
        }

    def save_scene_masks(self, scene_masks_data: Dict, filename: str = None) -> str:
        """保存场景掩码数据
        
        Args:
            scene_masks_data: 场景掩码数据
            filename: 保存文件名（可选）
            
        Returns:
            保存的文件路径
        """
        output_dir = Path(self.config['output_config']['save_path'])
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            prefix = self.config['output_config']['filename_prefix']
            version = self.config['dataset_config']['version']
            data_split = self.config['dataset_config']['data_split']
            num_scenes = len(scene_masks_data['scenes'])
            filename = f"{prefix}_{version}_{data_split}_{num_scenes}scenes.pkl"

        file_path = output_dir / filename

        with open(file_path, 'wb') as f:
            pickle.dump(scene_masks_data, f)

        return str(file_path)

    def load_scene_masks(self, file_path: str) -> Dict:
        """加载场景掩码数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            场景掩码数据
        """
        with open(file_path, 'rb') as f:
            scene_masks_data = pickle.load(f)

        return scene_masks_data

    def print_data_structure(self, data: Dict = None, file_path: str = None):
        """打印数据结构
        
        Args:
            data: 要打印的数据（可选）
            file_path: 数据文件路径（可选）
        """
        if data is None and file_path is None:
            # 查找最新的pkl文件
            output_dir = Path(self.config['output_config']['save_path'])
            pkl_files = list(output_dir.glob("*.pkl"))
            if pkl_files:
                latest_file = max(pkl_files, key=os.path.getctime)
                data = self.load_scene_masks(str(latest_file))
            else:
                return
        elif file_path is not None:
            data = self.load_scene_masks(file_path)

        print("\n" + "=" * 80)
        print("场景掩码数据结构")
        print("=" * 80)

        self._print_structure_recursive(data, indent=0)

    def _print_structure_recursive(self, obj, indent=0, key_name="root"):
        """递归打印数据结构"""
        prefix = "  " * indent

        if isinstance(obj, dict):
            print(f"{prefix}{key_name} (dict) - {len(obj)} keys:")
            for k, v in obj.items():
                if k == 'scenes' and isinstance(v, dict):
                    print(f"{prefix}  scenes (dict) - {len(v)} scenes:")
                    # 只显示第一个场景的详细结构
                    if v:
                        first_scene_key = list(v.keys())[0]
                        print(f"{prefix}    └── 示例场景 [{first_scene_key}]:")
                        self._print_structure_recursive(v[first_scene_key], indent + 3, "scene_data")
                else:
                    self._print_structure_recursive(v, indent + 1, k)
        elif isinstance(obj, list):
            print(f"{prefix}{key_name} (list) - {len(obj)} items")
            if obj and len(obj) > 0:
                print(f"{prefix}  └── 示例项目 [0]:")
                self._print_structure_recursive(obj[0], indent + 2, "item")
        elif isinstance(obj, np.ndarray):
            print(f"{prefix}{key_name} (ndarray) - shape: {obj.shape}, dtype: {obj.dtype}")
            if obj.size <= 10:
                print(f"{prefix}  └── 值: {obj.flatten()}")
            else:
                print(f"{prefix}  └── 前10个值: {obj.flatten()[:10]}")
        elif isinstance(obj, (str, int, float, bool)):
            sample_str = str(obj)
            if len(sample_str) > 50:
                sample_str = sample_str[:50] + "..."
            print(f"{prefix}{key_name} ({type(obj).__name__}) - {sample_str}")
        else:
            print(f"{prefix}{key_name} ({type(obj).__name__})")


def main():
    """主函数"""
    config_path = "../config/map_mask_config.yaml"

    # 创建掩码生成器
    generator = SceneMaskGenerator(config_path)

    # 生成场景掩码
    scene_masks_data = generator.generate_scene_masks()

    # 保存数据
    saved_file = generator.save_scene_masks(scene_masks_data)

    # 打印数据结构
    generator.print_data_structure(scene_masks_data)

    print(f"\n场景掩码生成完成！数据已保存到: {saved_file}")


if __name__ == "__main__":
    main()
