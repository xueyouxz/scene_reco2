#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取模型的latent space表示
将训练和测试数据的编码结果保存为JSON格式
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from minisom import MiniSom


# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.model_v2.tools.data_loader import load_config, ImprovedSceneDataset
from model.model_v2.autoencoder import create_autoencoder


class LatentSpaceExtractor:
    def __init__(self, config_path: str, model_path: str, output_dir: str, nuscenes_dataroot: str = None):
        self.config = load_config(config_path)
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)

        # 设置设备
        self.device = self._get_device()

        # 加载模型
        self.model = self._load_model()

        # 创建数据集
        self._create_datasets()


    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if self.config['device_config']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device_config']['gpu_id']}")
        else:
            device = torch.device('cpu')
        return device

    def _load_model(self) -> nn.Module:
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 使用保存的配置创建模型
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config

        # 创建模型
        model = create_autoencoder(model_config)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        # 获取模型信息
        model_info = model.get_model_info()
        print(f"模型加载成功:")
        print(f"  隐藏维度: {model_info['hidden_dimension']}")
        print(f"  总参数数: {model_info['total_parameters']:,}")

        return model

    def _create_datasets(self):
        """创建数据集和数据加载器"""
        # 训练数据集（包含训练和验证）
        self.train_dataset = ImprovedSceneDataset(
            data_path=self.config['data_config']['train_path'],
            config=self.config,
            mode='test'  # 使用test模式避免数据增强
        )

        # 测试数据集
        self.test_dataset = ImprovedSceneDataset(
            data_path=self.config['data_config']['test_path'],
            config=self.config,
            mode='test'
        )

        # 创建数据加载器
        batch_size = self.config['training_config']['batch_size']
        num_workers = self.config['device_config']['num_workers']

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,  # 保持顺序
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"数据集创建完成:")
        print(f"  训练集大小: {len(self.train_dataset)}")
        print(f"  测试集大小: {len(self.test_dataset)}")

    def extract_latent_space(self, data_loader: DataLoader,
                             dataset_name: str) -> Dict[str, List[float]]:

        print(f"\n提取{dataset_name}的latent space...")

        latent_dict = {}

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader,
                                                        desc=f"处理{dataset_name}")):
                # 解析批次数据
                if len(batch_data) == 3:
                    inputs, _, metadata = batch_data
                else:
                    inputs, _ = batch_data
                    metadata = {'scene_token': [f'sample_{i}'
                                                for i in range(len(inputs))]}

                inputs = inputs.to(self.device)

                # 获取编码表示
                # 注意：根据模型实现，encoder可能返回tuple
                if hasattr(self.model, 'encoder'):
                    encoder_output = self.model.encoder(inputs)

                    # 处理encoder返回tuple的情况（encoded, skip_features）
                    if isinstance(encoder_output, tuple):
                        encoded = encoder_output[0]
                    else:
                        encoded = encoder_output
                else:
                    # 使用整个模型的encode方法
                    encoded = self.model.encode(inputs)

                # 转换为CPU并转为列表
                encoded_np = encoded.cpu().numpy()

                # 保存每个样本
                batch_size = encoded_np.shape[0]
                for i in range(batch_size):
                    scene_token = metadata['scene_token'][i]
                    latent_vector = encoded_np[i].tolist()
                    latent_dict[scene_token] = latent_vector

        print(f"  提取完成，共{len(latent_dict)}个场景")

        return latent_dict


    def get_scene_name(self, scene_token: str) -> str:
        """通过scene_token获取场景名称"""
        if self.nusc is None:
            # 如果没有初始化nuScenes，返回默认命名
            return f"unknown-{scene_token[:8]}"
            
        try:
            # 使用nuScenes API获取场景信息
            scene = self.nusc.get('scene', scene_token)
            return scene['name']
        except Exception as e:
            print(f"获取场景名称失败: {e}")
            return f"unknown-{scene_token[:8]}"
    
    def apply_som(self, data: np.ndarray, map_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """使用SOM算法对数据进行降维
        
        Args:
            data: 高维数据 [n_samples, n_features]
            map_size: SOM网格大小 (width, height)
            
        Returns:
            降维后的数据 [n_samples, 2]
        """
        print(f"\n应用SOM算法进行降维 (网格大小: {map_size})...")
        
        # 获取数据维度
        n_samples, n_features = data.shape
        
        # 初始化SOM网络
        som = MiniSom(
            x=map_size[0], 
            y=map_size[1], 
            input_len=n_features, 
            sigma=1.0,
            learning_rate=0.5,
            neighborhood_function='gaussian', 
            random_seed=42
        )
        
        # 随机初始化权重
        som.random_weights_init(data)
        
        # 训练SOM
        print("训练SOM网络...")
        som.train_batch(
            data, 
            num_iteration=5000,
            verbose=True
        )
        
        # 获取每个样本在SOM网格中的位置
        som_coordinates = np.zeros((n_samples, 2))
        for i, x in enumerate(data):
            w = som.winner(x)
            som_coordinates[i, 0] = w[0]  # x坐标
            som_coordinates[i, 1] = w[1]  # y坐标
            
        print("SOM降维完成")
        return som_coordinates
            
    def save_dimension_reduction_results(self, scene_tokens: List[str], 
                                         tsne_results: np.ndarray,
                                         som_results: np.ndarray):
        """保存降维结果为指定格式的JSON"""
        output_path = self.output_dir / 'dimension_reduction.json'
        
        # 准备数据结构
        scenes_data = []
        for i, token in enumerate(scene_tokens):
            scene_name = self.get_scene_name(token)
            scenes_data.append({
                "scene_name": scene_name,
                "scene_token": token,
                "tsne_comp1": float(tsne_results[i, 0]),
                "tsne_comp2": float(tsne_results[i, 1]),
                "som_comp1": float(som_results[i, 0]),
                "som_comp2": float(som_results[i, 1])
            })
        
        # 构建最终JSON结构
        dimension_reduction_data = {
            "scene_counts": len(scenes_data),
            "scenes": scenes_data
        }
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dimension_reduction_data, f, indent=2, ensure_ascii=False)
            
        print(f"降维结果已保存到: {output_path}")
    
    def visualize_projection(self, train_latent: Dict, test_latent: Dict):
        print("\n生成latent space可视化（t-SNE和SOM）...")

        # 合并数据
        train_array = np.array(list(train_latent.values()))
        test_array = np.array(list(test_latent.values()))
        all_data = np.vstack([train_array, test_array])

        # 获取所有scene_token
        train_tokens = list(train_latent.keys())
        test_tokens = list(test_latent.keys())
        all_tokens = train_tokens + test_tokens

        # 创建图形布局：左侧t-SNE，右侧SOM
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # 1. t-SNE 降维
        print("执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(all_data)
        train_tsne = tsne_result[:len(train_array)]
        test_tsne = tsne_result[len(train_array):]

        # 绘制t-SNE结果
        axes[0].scatter(train_tsne[:, 0], train_tsne[:, 1], alpha=0.5,
                        s=10, c='blue', label='训练集')
        axes[0].scatter(test_tsne[:, 0], test_tsne[:, 1], alpha=0.5,
                        s=10, c='red', label='测试集')
        axes[0].set_xlabel('t-SNE 维度1')
        axes[0].set_ylabel('t-SNE 维度2')
        axes[0].set_title('t-SNE 降维可视化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. SOM 降维
        print("执行SOM降维...")
        som_result = self.apply_som(all_data)
        train_som = som_result[:len(train_array)]
        test_som = som_result[len(train_array):]

        # 绘制SOM结果
        axes[1].scatter(train_som[:, 0], train_som[:, 1], alpha=0.5,
                        s=20, c='blue', label='训练集')
        axes[1].scatter(test_som[:, 0], test_som[:, 1], alpha=0.5,
                        s=20, c='red', label='测试集')
        axes[1].set_xlabel('SOM 维度1')
        axes[1].set_ylabel('SOM 维度2')
        axes[1].set_title('SOM 降维可视化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        vis_path = self.output_dir / 'latent_space_visualization.png'
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"可视化已保存到: {vis_path}")
        
        # 保存降维结果为JSON
        self.save_dimension_reduction_results(all_tokens, tsne_result, som_result)

    def run(self):
        # 提取训练集的latent space
        train_latent = self.extract_latent_space(
            self.train_loader,
            "训练集"
        )

        # 提取测试集的latent space
        test_latent = self.extract_latent_space(
            self.test_loader,
            "测试集"
        )

        # 生成可视化
        self.visualize_projection(train_latent, test_latent)


        # 返回结果
        return train_latent, test_latent


def main():
    config_path = Path(__file__).resolve().parents[3] / 'config' / 'autoencoder_config_v2_3channels.yaml'
    model_path = '/home/zhangxueyou/PycharmProjects/scene_reco2/ckpt/checkpoints_3ch/best_model.pth'
    output = '/home/zhangxueyou/PycharmProjects/scene_reco2/output/latent_space_3channels'

    nuscenes_dataroot = '/home/public/nuscenes_datasets/nuscenes-trainval'
    
    # 创建提取器并运行
    extractor = LatentSpaceExtractor(
        config_path=config_path,
        model_path=model_path,
        output_dir=output,
        nuscenes_dataroot=nuscenes_dataroot
    )

    extractor.run()




if __name__ == "__main__":
    main()
