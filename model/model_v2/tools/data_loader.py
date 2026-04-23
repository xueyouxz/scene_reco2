import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config_loader import load_config


class ImprovedSceneDataset(Dataset):
    """改进的数据集，支持数据增强和更好的预处理"""

    def __init__(self, data_path: str, config: Dict, mode: str = 'train'):
        """
        初始化数据集
        Args:
            data_path: pkl数据文件路径
            config: 配置字典
            mode: 'train', 'val', or 'test'
        """
        self.data_path = data_path
        self.config = config
        self.mode = mode

        # 加载数据
        print(f"正在加载数据文件: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.scenes = list(self.data['scenes'].keys())
        print(f"加载了 {len(self.scenes)} 个场景")

        # 数据处理配置
        data_config = config['data_processing']
        
        # 栅格尺寸配置
        self.grid_height = data_config['grid_height']
        self.grid_width = data_config['grid_width']
        self.input_height = self.grid_height
        self.input_width = self.grid_width
        
        # 通道配置
        self.use_ego_trajectory = data_config['use_ego_trajectory']
        self.base_channels = data_config['base_channels']
        self.input_channels = self.base_channels + (1 if self.use_ego_trajectory else 0)
        
        # 通道映射配置
        self.channel_names = ['divider', 'drivable_area', 'ped_crossing']
        if self.use_ego_trajectory:
            self.channel_names = ['ego_trajectory'] + self.channel_names
            
        print(f"栅格尺寸: {self.grid_height}x{self.grid_width}")
        print(f"使用自车轨迹: {self.use_ego_trajectory}")
        print(f"输入通道数: {self.input_channels}")
        print(f"通道名称: {self.channel_names}")

        # 创建数据增强管道
        self.transform = self._create_transform()

        # 计算类别权重（用于损失函数）
        if mode == 'train':
            self._compute_class_weights()

    def _create_transform(self):
        """创建数据增强管道 - 根据栅格尺寸动态调整增强策略"""
        # 判断栅格是否为正方形
        is_square_grid = (self.grid_height == self.grid_width)
        print(f"栅格是否为正方形: {is_square_grid} ({self.grid_height}x{self.grid_width})")
        
        if self.mode == 'train':
            # 基础变换列表
            transforms = []
            
            # 基础翻转操作（所有情况下都适用）
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
            
            # 根据栅格形状决定是否添加旋转操作
            if is_square_grid:
                print("正方形栅格：启用旋转增强")
                # 正方形栅格可以安全地进行旋转操作
                transforms.extend([
                    # 90度倍数旋转（保持网格结构）
                    A.RandomRotate90(p=0.5),
                    
                    # 小角度旋转
                    A.Rotate(
                        limit=15,  # 限制旋转角度为±15度
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.4
                    ),
                    
                    # 包含旋转的平移缩放变换
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=5,  # 正方形时允许小角度旋转
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5
                    ),
                ])
            else:
                print("矩形栅格：使用原始增强策略（无旋转）")
                # 矩形栅格避免旋转，使用其他增强方式
                transforms.extend([
                    # 平移、缩放（不包含旋转）
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=0,  # 矩形时禁用旋转
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5
                    ),
                ])
            
            # 通用增强操作（所有情况下都适用）
            transforms.extend([
                # 弹性变换（模拟道路变形）
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    alpha_affine=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.3
                ),

                # 透视变换（模拟视角变化）
                A.Perspective(
                    scale=(0.05, 0.1),
                    keep_size=True,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=0,
                    p=0.3
                ),

                # 网格失真（增加鲁棒性）
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.3
                ),

                # 噪声（提高鲁棒性）
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),

                # 确保输出尺寸正确
                A.Resize(
                    height=self.input_height,
                    width=self.input_width,
                    interpolation=cv2.INTER_LINEAR,
                    always_apply=True
                ),
            ])
            
            transform = A.Compose(transforms)
        else:
            # 验证和测试模式不使用数据增强
            transform = A.Compose([
                A.Resize(
                    height=self.input_height,
                    width=self.input_width,
                    interpolation=cv2.INTER_LINEAR,
                    always_apply=True
                ),
            ])

        return transform

    def _compute_class_weights(self):
        """计算每个通道的类别权重"""
        print("计算类别权重...")
        channel_pos_ratios = np.zeros(self.input_channels)

        # 采样计算
        sample_size = min(100, len(self.scenes))
        for i in range(sample_size):
            scene_data = self.data['scenes'][self.scenes[i]]
            masks = scene_data['masks']

            for c, channel_name in enumerate(self.channel_names):
                if channel_name in masks:
                    mask = masks[channel_name]
                    pos_ratio = np.mean(mask > 0.5)
                    channel_pos_ratios[c] += pos_ratio
                else:
                    print(f"警告：通道 '{channel_name}' 在数据中不存在")

        channel_pos_ratios /= sample_size

        # 计算权重（逆频率加权）
        self.class_weights = 1.0 / (channel_pos_ratios + 0.01)
        self.class_weights /= self.class_weights.sum()

        print(f"使用通道: {self.channel_names}")
        print(f"通道正样本比例: {channel_pos_ratios}")
        print(f"类别权重: {self.class_weights}")

    def _resize_data(self, data: np.ndarray) -> np.ndarray:
        """
        使用改进的插值方法调整数据尺寸
        """
        C, H, W = data.shape
        resized_data = np.zeros((C, self.input_height, self.input_width), dtype=np.float32)

        for c in range(C):
            # 对于二值掩码，先使用双线性插值，然后二值化
            resized = cv2.resize(
                data[c].astype(np.float32),
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR
            )
            # 软二值化（保留一些边界信息）
            resized = 1.0 / (1.0 + np.exp(-10 * (resized - 0.5)))
            resized_data[c] = resized

        return resized_data

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        获取单个数据样本
        返回: (输入, 目标, 元数据)
        """
        scene_token = self.scenes[idx]
        scene_data = self.data['scenes'][scene_token]

        # 提取掩码数据 - 根据配置动态选择通道
        masks = scene_data['masks']
        channel_data_list = []
        
        for channel_name in self.channel_names:
            if channel_name in masks:
                channel_data_list.append(masks[channel_name])
            else:
                # 如果某个通道不存在，创建全零掩码
                print(f"警告：通道 '{channel_name}' 不存在，使用零掩码")
                dummy_mask = np.zeros_like(next(iter(masks.values())))
                channel_data_list.append(dummy_mask)
        
        multi_channel_data = np.stack(channel_data_list, axis=0)

        # 调整尺寸
        resized_data = self._resize_data(multi_channel_data)

        # 转换为HWC格式用于增强
        data_hwc = resized_data.transpose(1, 2, 0)  # (H, W, C)

        # 确保数据类型正确
        data_hwc = data_hwc.astype(np.float32)

        # 应用数据增强
        if self.mode == 'train':
            try:
                # albumentations期望图像是uint8或float32
                # 对于二值mask，我们使用float32
                augmented = self.transform(image=data_hwc)
                data_hwc = augmented['image']

                # 确保数据在[0,1]范围内
                data_hwc = np.clip(data_hwc, 0, 1)

                # 再次确保尺寸正确（双重保险）
                if data_hwc.shape != (self.input_height, self.input_width, self.input_channels):
                    print(f"警告：增强后尺寸不正确 {data_hwc.shape}，进行修正...")
                    data_hwc = cv2.resize(data_hwc, (self.input_width, self.input_height))
                    if len(data_hwc.shape) == 2:  # 如果变成2D，恢复通道维度
                        data_hwc = np.expand_dims(data_hwc, axis=-1)

            except Exception as e:
                print(f"数据增强失败: {e}，使用原始数据")
                # 如果增强失败，使用原始数据
                pass
        else:
            # 验证和测试模式也应用transform（只有Resize）
            augmented = self.transform(image=data_hwc)
            data_hwc = augmented['image']
            data_hwc = np.clip(data_hwc, 0, 1)

        # 最终尺寸验证
        assert data_hwc.shape == (self.input_height, self.input_width, self.input_channels), \
            f"输出尺寸错误: {data_hwc.shape}，期望: ({self.input_height}, {self.input_width}, {self.input_channels})"

        # 转换为张量
        data_tensor = torch.from_numpy(data_hwc).float()

        # 元数据
        metadata = {
            'scene_token': scene_token,
            'original_shape': multi_channel_data.shape
        }

        return data_tensor, data_tensor, metadata


def collate_fn(batch):
    """自定义collate函数，处理可能的尺寸不一致问题"""
    inputs = []
    targets = []
    metadata = []

    for item in batch:
        inp, tgt, meta = item

        # 验证尺寸
        if inp.shape != batch[0][0].shape:
            print(f"警告：批次中存在尺寸不一致: {inp.shape} vs {batch[0][0].shape}")
            # 可以选择跳过这个样本或调整尺寸
            continue

        inputs.append(inp)
        targets.append(tgt)
        metadata.append(meta)

    # 堆叠张量
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    # 合并元数据
    merged_metadata = {}
    for key in metadata[0].keys():
        merged_metadata[key] = [m[key] for m in metadata]

    return inputs, targets, merged_metadata


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建改进的数据加载器"""

    # 训练数据集
    train_dataset = ImprovedSceneDataset(
        data_path=config['data_config']['train_path'],
        config=config,
        mode='train'
    )

    # 划分训练和验证集
    train_size = int(config['data_config']['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )

    # 验证数据集（无增强）
    val_dataset = ImprovedSceneDataset(
        data_path=config['data_config']['train_path'],
        config=config,
        mode='val'
    )
    # 注意：这里需要正确设置val_subset的indices
    val_indices = val_subset.indices
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # 测试数据集
    test_dataset = ImprovedSceneDataset(
        data_path=config['data_config']['test_path'],
        config=config,
        mode='test'
    )

    # 创建数据加载器
    batch_size = config['training_config']['batch_size']
    num_workers = config['device_config']['num_workers']
    pin_memory = config['device_config']['pin_memory']

    # 使用自定义collate函数
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,  # 使用自定义collate函数
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,  # 使用自定义collate函数
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn  # 使用自定义collate函数
    )

    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 验证数据加载器
    print("验证数据加载器...")
    try:
        for i, (inputs, targets, metadata) in enumerate(train_loader):
            print(f"批次 {i}: 输入形状 {inputs.shape}, 目标形状 {targets.shape}")
            
            # 动态计算期望的通道数
            data_config = config['data_processing']
            expected_channels = data_config['base_channels'] + (1 if data_config['use_ego_trajectory'] else 0)
            expected_shape = (data_config['grid_height'], data_config['grid_width'], expected_channels)
            
            assert inputs.shape[1:] == expected_shape, \
                f"数据形状不正确: {inputs.shape}, 期望: (batch_size, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})"
            if i >= 2:  # 只检查前几个批次
                break
        print("数据加载器验证通过！")
    except Exception as e:
        print(f"数据加载器验证失败: {e}")
        raise

    return train_loader, val_loader, test_loader
