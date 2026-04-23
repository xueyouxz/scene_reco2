import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

from config_loader import load_config


class SceneDataset(Dataset):
    def __init__(self, data_path: str, config: Dict):
        """
        初始化数据集
        
        Args:
            data_path: pkl数据文件路径
            config: 配置字典
        """
        self.data_path = data_path
        self.config = config
        
        # 加载数据
        print(f"正在加载数据文件: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 提取场景列表
        self.scenes = list(self.data['scenes'].keys())
        print(f"加载了 {len(self.scenes)} 个场景")
        
        # 获取数据处理配置
        self.input_channels = config['data_processing']['input_channels']
        self.input_height = config['data_processing']['input_height']
        self.input_width = config['data_processing']['input_width']
        self.normalize = config['data_processing']['normalize']
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            输入张量和目标张量（自编码器中输入等于目标）
        """
        scene_token = self.scenes[idx]
        scene_data = self.data['scenes'][scene_token]
        
        # 提取4个通道的mask数据
        masks = scene_data['masks']
        
        # 按指定顺序组织通道：
        # 通道0: 自车轨迹 (ego_trajectory)
        # 通道1: 车道线/分隔线 (divider) 
        # 通道2: 驾驶区域 (drivable_area)
        # 通道3: 人行横道 (ped_crossing)
        multi_channel_data = np.stack([
            masks['ego_trajectory'],
            masks['divider'], 
            masks['drivable_area'],
            masks['ped_crossing']
        ], axis=0)  # shape: (4, H, W)
        
        # 调整到目标尺寸 (4, 700, 400)
        resized_data = self._resize_data(multi_channel_data)
        
        # 转换为torch张量并调整维度顺序为 (H, W, C)
        data_tensor = torch.from_numpy(resized_data).float()
        data_tensor = data_tensor.permute(1, 2, 0)  # (H, W, C)
        
        # 二值化处理：确保数据为0或1
        # 对于二值栅格数据，通常大于0.5的设为1，否则为0
        data_tensor = (data_tensor > 0.5).float()
        
        # 对于二值数据，不需要除以255标准化，直接使用0和1
        
        # 对于自编码器，输入和目标相同
        return data_tensor, data_tensor
    
    def _resize_data(self, data: np.ndarray) -> np.ndarray:
        """
        调整数据尺寸到目标尺寸
        
        Args:
            data: 输入数据 (C, H, W)
            
        Returns:
            调整后的数据 (C, target_H, target_W)
        """
        import cv2
        
        C, H, W = data.shape
        resized_data = np.zeros((C, self.input_height, self.input_width), dtype=data.dtype)
        
        for c in range(C):
            resized_data[c] = cv2.resize(
                data[c], 
                (self.input_width, self.input_height),  # cv2.resize expects (width, height)
                interpolation=cv2.INTER_NEAREST
            )
        
        return resized_data


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建简化的数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        训练、验证、测试数据加载器
    """
    # 加载训练数据
    train_dataset = SceneDataset(
        data_path=config['data_config']['train_path'],
        config=config
    )
    
    # 划分训练和验证集
    train_size = int(config['data_config']['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )
    
    # 为验证集创建新的数据集实例
    val_dataset = SceneDataset(
        data_path=config['data_config']['train_path'],
        config=config
    )
    val_subset.dataset = val_dataset
    
    # 加载测试数据
    test_dataset = SceneDataset(
        data_path=config['data_config']['test_path'],
        config=config
    )
    
    # 创建数据加载器
    batch_size = config['training_config']['batch_size']
    num_workers = config['device_config']['num_workers']
    pin_memory = config['device_config']['pin_memory']
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


