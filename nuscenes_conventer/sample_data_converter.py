import logging
import os
import pickle
from datetime import datetime
from os import path as osp
from pathlib import Path
from typing import Dict

import mmcv
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm

from config_loader import load_config
from nuscenes_conventer.map_utils import NuscMapExtractor


class NuScenesDataConverter:
    # 类别名称映射
    NAME_MAPPING = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    def __init__(self, config_path: str):

        self.config = load_config(config_path)

    def convert(self):
        """执行数据转换"""
        version = self.config['conversion_params']['version']
        if not version.startswith('v1.0-'):
            version = f'v1.0-{version}'

        # 获取路径配置
        version_key = version.replace('v1.0-', '')
        dataset_config = self.config['dataset_paths'][version_key]
        root_path = dataset_config['root_path']
        can_bus_path = dataset_config['can_bus_path']

        # 输出配置
        output_dir = Path(self.config['conversion_params']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建转换信息
        self._create_nuscenes_infos(
            root_path=root_path,
            out_path=str(output_dir),
            can_bus_root_path=can_bus_path,
            version=version
        )

    def _create_nuscenes_infos(self, root_path, out_path, can_bus_root_path, version):

        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

        # 初始化地图提取器
        roi_size = tuple(self.config['processing_params']['roi_size'])
        nusc_map_extractor = NuscMapExtractor(root_path, roi_size)

        # 初始化CAN总线API
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)

        # 初始化预测辅助器
        predict_helper = PredictHelper(nusc)

        # 获取数据集转换配置
        datasets_to_convert = self.config['conversion_params']['datasets_to_convert']

        # 获取场景分割
        from nuscenes.utils import splits
        if version == 'v1.0-trainval':
            train_scenes = splits.train if 'train' in datasets_to_convert else []
            val_scenes = splits.val if 'val' in datasets_to_convert else []
        elif version == 'v1.0-test':
            train_scenes = splits.test if 'test' in datasets_to_convert else []
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train if 'train' in datasets_to_convert else []
            val_scenes = splits.mini_val if 'val' in datasets_to_convert else []
        else:
            raise ValueError(f'Unknown version: {version}')

        # 如果没有要处理的场景，直接返回空列表
        if not train_scenes and not val_scenes:
            print(f"没有需要处理的数据集，请检查配置文件中的datasets_to_convert设置")
            train_infos, val_infos = [], []
            metadata = self._build_metadata(version)
            self._save_infos(out_path, train_infos, val_infos, metadata, version)
            return

        # 过滤可用场景
        available_scenes = self._get_available_scenes(nusc)
        available_scene_names = [s['name'] for s in available_scenes]

        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

        train_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in train_scenes
        ]) if train_scenes else set()

        val_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in val_scenes
        ]) if val_scenes else set()

        # 填充训练和验证信息
        train_infos, val_infos = self._fill_trainval_infos(
            nusc, nusc_map_extractor, nusc_can_bus, predict_helper,
            train_scenes, val_scenes, 'test' in version
        )

        # 构建元数据
        metadata = self._build_metadata(version)

        # 保存信息文件
        self._save_infos(out_path, train_infos, val_infos, metadata, version)

    def _get_available_scenes(self, nusc):
        """获取可用场景"""
        available_scenes = []

        for scene in nusc.scene:
            scene_token = scene['token']
            scene_rec = nusc.get('scene', scene_token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

            has_more_frames = True
            scene_not_exist = False

            while has_more_frames:
                lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
                lidar_path = str(lidar_path)

                if os.getcwd() in lidar_path:
                    lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]

                if not mmcv.is_filepath(lidar_path):
                    scene_not_exist = True
                    break
                else:
                    break

            if not scene_not_exist:
                available_scenes.append(scene)

        return available_scenes

    def _fill_trainval_infos(self, nusc, nusc_map_extractor, nusc_can_bus,
                             predict_helper, train_scenes, val_scenes, test=False):
        """填充训练和验证信息"""

        train_nusc_infos = []
        val_nusc_infos = []

        # 获取处理参数
        max_sweeps = self.config['processing_params']['max_sweeps']
        fut_ts = self.config['processing_params']['future_timesteps']['agents']
        ego_fut_ts = self.config['processing_params']['future_timesteps']['ego']

        # 构建类别索引
        cat2idx = {}
        for idx, dic in enumerate(nusc.category):
            cat2idx[dic['name']] = idx

        # 处理每个样本
        samples_to_process = nusc.sample
        need_train = len(train_scenes) > 0
        need_val = len(val_scenes) > 0

        for sample in tqdm(samples_to_process, desc="Processing samples"):
            scene_token = sample['scene_token']

            # 检查该样本是否需要处理
            is_train_sample = scene_token in train_scenes
            is_val_sample = scene_token in val_scenes

            if (need_train and not is_train_sample) and (need_val and not is_val_sample):
                continue

            info = self._process_sample(
                sample, nusc, nusc_map_extractor, nusc_can_bus,
                predict_helper, max_sweeps, fut_ts, ego_fut_ts, test
            )

            if info is None:
                continue

            # 分配到训练或验证集
            if need_train and is_train_sample:
                train_nusc_infos.append(info)
            elif need_val and is_val_sample:
                val_nusc_infos.append(info)

        return train_nusc_infos, val_nusc_infos

    def _process_sample(self, sample, nusc, nusc_map_extractor, nusc_can_bus,
                        predict_helper, max_sweeps, fut_ts, ego_fut_ts, test):
        """处理单个样本"""

        # 获取基本信息
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        if not osp.isfile(lidar_path):
            return None

        # 构建信息字典
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'map_location': map_location,
        }

        # 获取变换矩阵
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # 提取地图标注
        info['map_annos'] = self._extract_map_annos(
            info, nusc_map_extractor, map_location
        )

        # 提取相机信息
        info['cams'] = self._extract_camera_info(
            sample, nusc, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat
        )

        # 提取sweeps信息
        info['sweeps'] = self._extract_sweeps(
            sample, nusc, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, max_sweeps
        )

        # 如果不是测试模式，提取标注信息
        if not test:
            annos = self._extract_annotations(
                sample, nusc, nusc_can_bus, predict_helper,
                boxes, l2e_r_mat, e2g_r_mat, fut_ts, ego_fut_ts,
                pose_record, cs_record
            )
            info.update(annos)

        return info

    def _extract_map_annos(self, info, nusc_map_extractor, map_location):
        """提取地图标注"""
        # 构建变换矩阵
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])

        lidar2global = ego2global @ lidar2ego

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)

        # 获取地图几何
        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation)

        # 转换为标注格式
        MAP_CLASSES = ('ped_crossing', 'divider', 'boundary')
        vectors = {}

        for cls, geom_list in map_geoms.items():
            if cls in MAP_CLASSES:
                label = MAP_CLASSES.index(cls)
                vectors[label] = []
                for geom in geom_list:
                    if hasattr(geom, 'coords'):
                        line = np.array(geom.coords)
                        vectors[label].append(line)

        return vectors

    def _extract_camera_info(self, sample, nusc, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
        """提取相机信息"""
        camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ]

        cams = {}
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = self._obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            cams[cam] = cam_info

        return cams

    def _extract_sweeps(self, sample, nusc, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, max_sweeps):
        """提取sweeps信息"""
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []

        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = self._obtain_sensor2top(
                    nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar'
                )
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break

        return sweeps

    def _extract_annotations(self, sample, nusc, nusc_can_bus, predict_helper,
                             boxes, l2e_r_mat, e2g_r_mat, fut_ts, ego_fut_ts,
                             pose_record, cs_record):
        """提取标注信息"""
        annotations = [
            nusc.get('sample_annotation', token)
            for token in sample['anns']
        ]

        # 提取基本信息
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        # 提取速度
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        # 处理类别名称
        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in self.NAME_MAPPING:
                names[i] = self.NAME_MAPPING[names[i]]
        names = np.array(names)

        # 有效标志
        valid_flag = np.array(
            [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
             for anno in annotations],
            dtype=bool
        ).reshape(-1)

        # 构建GT boxes
        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

        # 实例索引
        instance_inds = [nusc.getind('instance', anno['instance_token'])
                         for anno in annotations]

        # 提取未来轨迹
        gt_fut_trajs, gt_fut_masks = self._extract_future_trajectories(
            annotations, predict_helper, sample['token'], boxes, fut_ts
        )

        # 提取自车未来轨迹
        ego_fut_trajs, ego_fut_masks, command = self._extract_ego_future(
            sample, nusc, ego_fut_ts, pose_record, cs_record
        )

        # 获取自车状态
        ego_status = self._get_ego_status(nusc, nusc_can_bus, sample)

        return {
            'gt_boxes': gt_boxes,
            'gt_names': names,
            'gt_velocity': velocity.reshape(-1, 2),
            'num_lidar_pts': np.array([a['num_lidar_pts'] for a in annotations]),
            'num_radar_pts': np.array([a['num_radar_pts'] for a in annotations]),
            'valid_flag': valid_flag,
            'instance_inds': instance_inds,
            'gt_agent_fut_trajs': gt_fut_trajs.astype(np.float32),
            'gt_agent_fut_masks': gt_fut_masks.astype(np.float32),
            'gt_ego_fut_trajs': ego_fut_trajs[:, :2].astype(np.float32),
            'gt_ego_fut_masks': ego_fut_masks[1:].astype(np.float32),
            'gt_ego_fut_cmd': command.astype(np.float32),
            'ego_status': ego_status
        }

    def _extract_future_trajectories(self, annotations, predict_helper,
                                     sample_token, boxes, fut_ts):
        """提取未来轨迹"""
        num_box = len(boxes)
        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
        gt_fut_masks = np.zeros((num_box, fut_ts))

        for i, anno in enumerate(annotations):
            instance_token = anno['instance_token']
            fut_traj_local = predict_helper.get_future_for_agent(
                instance_token,
                sample_token,
                seconds=fut_ts / 2,
                in_agent_frame=True
            )

            if fut_traj_local.shape[0] > 0:
                box = boxes[i]
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_traj_scene = convert_local_coords_to_global(fut_traj_local, trans, rot)
                valid_step = fut_traj_scene.shape[0]
                gt_fut_trajs[i, 0] = fut_traj_scene[0] - box.center[:2]
                gt_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1]
                gt_fut_masks[i, :valid_step] = 1

        return gt_fut_trajs, gt_fut_masks

    def _extract_ego_future(self, sample, nusc, ego_fut_ts, pose_record, cs_record):
        """提取自车未来轨迹"""
        ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
        ego_fut_masks = np.zeros((ego_fut_ts + 1))
        sample_cur = sample

        for i in range(ego_fut_ts + 1):
            pose_mat = self._get_global_sensor_pose(sample_cur, nusc)
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1

            if sample_cur['next'] == '':
                ego_fut_trajs[i + 1:] = ego_fut_trajs[i]
                break
            else:
                sample_cur = nusc.get('sample', sample_cur['next'])

        # 坐标变换
        ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
        rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

        ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

        # 驾驶指令
        if ego_fut_trajs[-1][0] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif ego_fut_trajs[-1][0] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight

        ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        return ego_fut_trajs, ego_fut_masks, command

    def _get_ego_status(self, nusc, nusc_can_bus, sample):
        """获取自车状态"""
        ego_status = []
        ref_scene = nusc.get("scene", sample['scene_token'])

        try:
            pose_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'pose')
            steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            ref_utime = sample['timestamp']

            pose_index = self._locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = self._locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]

            ego_status.extend(pose_data["accel"])
            ego_status.extend(pose_data["rotation_rate"])
            ego_status.extend(pose_data["vel"])
            ego_status.append(steer_data["value"])
        except:
            ego_status = [0] * 10

        return np.array(ego_status).astype(np.float32)

    def _locate_message(self, utimes, utime):
        """定位消息"""
        i = np.searchsorted(utimes, utime)
        if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
            i -= 1
        return i

    def _get_global_sensor_pose(self, rec, nusc):
        """获取全局传感器姿态"""
        lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

        ego2global = transform_matrix(
            pose_record["translation"],
            Quaternion(pose_record["rotation"]),
            inverse=False
        )
        sensor2ego = transform_matrix(
            cs_record["translation"],
            Quaternion(cs_record["rotation"]),
            inverse=False
        )
        pose = ego2global.dot(sensor2ego)

        return pose

    def _obtain_sensor2top(self, nusc, sensor_token, l2e_t, l2e_r_mat,
                           e2g_t, e2g_r_mat, sensor_type='lidar'):
        """获取传感器到顶部激光雷达的变换"""
        sd_rec = nusc.get('sample_data', sensor_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        data_path = str(nusc.get_sample_data_path(sd_rec['token']))

        if os.getcwd() in data_path:
            data_path = data_path.split(f'{os.getcwd()}/')[-1]

        sweep = {
            'data_path': data_path,
            'type': sensor_type,
            'sample_data_token': sd_rec['token'],
            'sensor2ego_translation': cs_record['translation'],
            'sensor2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sd_rec['timestamp']
        }

        l2e_r_s = sweep['sensor2ego_rotation']
        l2e_t_s = sweep['sensor2ego_translation']
        e2g_r_s = sweep['ego2global_rotation']
        e2g_t_s = sweep['ego2global_translation']

        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T -= e2g_t @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

        sweep['sensor2lidar_rotation'] = R.T
        sweep['sensor2lidar_translation'] = T

        return sweep

    def _build_metadata(self, version):
        """
        构建元数据
        todo: 向元数据中添加版本，转换的数据集（val/train）,ROI，sample数量字段
        """
        metadata = {
            'version': version,
            'config': self.config,
        }
        return metadata

    def _save_infos(self, out_path, train_infos, val_infos, metadata, version):
        """保存信息文件"""
        prefix = self.config['conversion_params']['info_prefix']
        extra_tag = self.config['conversion_params']['extra_tag']
        datasets_to_convert = self.config['conversion_params']['datasets_to_convert']

        # 构建文件名
        if extra_tag:
            prefix = f"{prefix}_{extra_tag}"

        if 'test' in version:
            if 'test' in datasets_to_convert and len(train_infos) > 0:
                data = dict(infos=train_infos, metadata=metadata)
                filename = f"{prefix}_{version}_test_{len(train_infos)}samples.pkl"
                info_path = osp.join(out_path, filename)
                print(f"保存测试数据集信息到：{info_path}")
                with open(info_path, 'wb') as f:
                    pickle.dump(data, f)
        else:
            if 'train' in datasets_to_convert and len(train_infos) > 0:
                data = dict(infos=train_infos, metadata=metadata)
                filename = f"{prefix}_{version}_train_{len(train_infos)}samples.pkl"
                info_path = osp.join(out_path, filename)
                print(f"保存训练数据集信息到：{info_path}")
                with open(info_path, 'wb') as f:
                    pickle.dump(data, f)

            if 'val' in datasets_to_convert and len(val_infos) > 0:
                data = dict(infos=val_infos, metadata=metadata)
                filename = f"{prefix}_{version}_val_{len(val_infos)}samples.pkl"
                info_path = osp.join(out_path, filename)
                print(f"保存验证数据集信息到：{info_path}")
                with open(info_path, 'wb') as f:
                    pickle.dump(data, f)

    def print_data_structure(self, pkl_file_path: str = None):
        """
        打印数据结构
        """
        if pkl_file_path is None:
            # 查找最新的pkl文件
            output_dir = Path(self.config['conversion_params']['output_dir'])
            pkl_files = sorted(output_dir.glob("*.pkl"), key=os.path.getctime, reverse=True)
            if not pkl_files:
                return
            pkl_file_path = pkl_files[0]

        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        print("\n" + "=" * 80)
        print(f"Data Structure for: {Path(pkl_file_path).name}")
        print("=" * 80)

        self._print_structure_recursive(data, indent=0)

    def _print_structure_recursive(self, obj, indent=0, key_name="root", sample_idx=0):
        """递归打印数据结构"""
        prefix = "  " * indent

        if isinstance(obj, dict):
            print(f"{prefix}{key_name} (dict) - {len(obj)} keys:")
            for k, v in obj.items():
                self._print_structure_recursive(v, indent + 1, k, sample_idx)

        elif isinstance(obj, list):
            print(f"{prefix}{key_name} (list) - {len(obj)} items")
            if len(obj) > 0:
                print(f"{prefix}  └── Sample item [0]:")
                self._print_structure_recursive(obj[0], indent + 2, "item", 0)

        elif isinstance(obj, np.ndarray):
            print(f"{prefix}{key_name} (ndarray) - shape: {obj.shape}, dtype: {obj.dtype}")
            if obj.size > 0 and obj.size <= 10:
                print(f"{prefix}  └── values: {obj.flatten()[:10]}")

        elif isinstance(obj, (str, int, float, bool)):
            sample_str = str(obj)
            if len(sample_str) > 50:
                sample_str = sample_str[:50] + "..."
            print(f"{prefix}{key_name} ({type(obj).__name__}) - example: {sample_str}")

        else:
            print(f"{prefix}{key_name} ({type(obj).__name__})")


def main():
    converter = NuScenesDataConverter('../config/sample_config.yaml')
    pkl_path = "./data/nuscenes_infos/nuscenes_vectorized_v1.0-trainval_train_700samples.pkl"
    if os.path.exists(pkl_path):
        converter.print_data_structure(pkl_path)
    else:
        converter.convert()
        converter.print_data_structure()


if __name__ == "__main__":
    main()
