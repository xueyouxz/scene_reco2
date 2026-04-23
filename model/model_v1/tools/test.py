#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化推理测试脚本
实现模型测试和结果可视化
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，将跳过图像可视化功能")

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.model_v1.tools.data_loader import create_data_loaders, load_config
from model.model_v1.autoencoder import create_autoencoder


class AutoencoderInference:
    """简化自编码器推理器"""

    def __init__(self, config: Dict, model_path: str):
        """
        初始化推理器
        
        Args:
            config: 配置字典
            model_path: 模型文件路径
        """
        self.config = config
        self.device = self._get_device()

        # 创建结果目录
        self._create_directories()

        # 加载模型
        self.model = self._load_model(model_path)

        # 创建数据加载器
        _, _, self.test_loader = create_data_loaders(config)

        print(f"测试数据集大小: {len(self.test_loader.dataset)}")

    def _get_device(self) -> torch.device:
        """获取设备"""
        if self.config['device_config']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device_config']['gpu_id']}")
            print(f"使用GPU: {device}")
        else:
            device = torch.device('cpu')
            print("使用CPU")
        return device

    def _create_directories(self):
        """创建必要的目录"""
        if self.config['visualization_config']['test_visualization']['enabled']:
            results_dir = self.config['visualization_config']['test_visualization']['results_dir']
            Path(results_dir).mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        print(f"加载模型: {model_path}")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 创建模型
        if 'config' in checkpoint:
            # 使用保存的配置创建模型
            model_config = checkpoint['config']
            model = create_autoencoder(model_config)
        else:
            # 使用当前配置创建模型
            model = create_autoencoder(self.config)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        print(f"模型加载完成")
        print(f"模型信息: {model.get_model_info()}")

        return model

    def visualize_reconstruction_results(self, num_samples: int = None):
        """可视化重建结果"""
        if not self.config['visualization_config']['test_visualization']['enabled']:
            print("测试可视化未启用，跳过...")
            return

        if num_samples is None:
            num_samples = self.config['visualization_config']['test_visualization']['num_test_samples']

        results_dir = Path(self.config['visualization_config']['test_visualization']['results_dir'])

        self.model.eval()
        sample_count = 0

        reconstruction_data = []
        input_data_list = []
        reconstructed_data_list = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if sample_count >= num_samples:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                reconstructed = self.model(inputs)

                # 转换为numpy
                inputs_np = inputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()

                # 保存每个样本的数据
                batch_size = inputs_np.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    sample_data = {
                        'input': inputs_np[i],
                        'target': targets_np[i],
                        'reconstructed': reconstructed_np[i],
                        'sample_id': sample_count
                    }
                    reconstruction_data.append(sample_data)

                    # 保存用于可视化的数据
                    input_data_list.append(inputs_np[i])
                    reconstructed_data_list.append(reconstructed_np[i])

                    # 创建详细的可视化图像
                    if MATPLOTLIB_AVAILABLE:
                        print(f"生成样本 {sample_count} 的详细可视化...")
                        viz_file = self.create_detailed_visualization(
                            inputs_np[i],
                            reconstructed_np[i],
                            sample_count,
                            results_dir
                        )
                        if viz_file:
                            print(f"详细可视化已保存到: {viz_file}")
                    else:
                        print(f"跳过样本 {sample_count} 的图像可视化（matplotlib不可用）")
                    reconstructed_binary = (reconstructed_np[i] > 0.5).astype(np.float32)
                    inputs_binary = (inputs_np[i] > 0.5).astype(np.float32)
                    accuracy_per_channel = np.mean(reconstructed_binary == inputs_binary, axis=(0, 1))
                    mse_per_channel = np.mean((inputs_np[i] - reconstructed_np[i]) ** 2, axis=(0, 1))

                    channel_names = ['ego_trajectory', 'divider', 'drivable_area', 'ped_crossing']
                    print(f"scene {sample_count}:")
                    for ch, (name, acc, mse) in enumerate(zip(channel_names, accuracy_per_channel, mse_per_channel)):
                        print(f"  {name}: acc={acc:.3f}, MSE={mse:.6f}")

                    sample_count += 1

        # 创建多样本对比网格
        if input_data_list and MATPLOTLIB_AVAILABLE:
            print("生成多样本对比网格...")
            comparison_file = self.create_comparison_grid(
                input_data_list,
                reconstructed_data_list,
                results_dir,
                max_samples=min(4, len(input_data_list))
            )
            if comparison_file:
                print(f"对比网格已保存到: {comparison_file}")
        elif input_data_list and not MATPLOTLIB_AVAILABLE:
            print("跳过对比网格生成（matplotlib不可用）")

        # 保存重建数据
        reconstruction_path = results_dir / 'reconstruction_results.npz'
        np.savez_compressed(
            reconstruction_path,
            **{f'sample_{i}': data for i, data in enumerate(reconstruction_data)}
        )
        print(f"重建结果已保存到: {reconstruction_path}")

        print(f"已生成 {sample_count} 个重建结果")

    def run_inference(self):
        self.visualize_reconstruction_results()

    def create_channel_overlay_visualization(self, data: np.ndarray,

                                             is_binary: bool = True) -> np.ndarray:
        """
        创建多通道叠加可视化
        
        Args:
            data: 输入数据 (H, W, C) 其中C=4
            is_binary: 是否为二值数据
            
        Returns:
            RGB图像数组
        """
        # 定义每个通道的颜色
        channel_colors = {
            0: [1.0, 0.0, 0.0],  # 自车轨迹 - 红色
            1: [0.0, 1.0, 0.0],  # 车道线 - 绿色
            2: [0.0, 0.0, 1.0],  # 驾驶区域 - 蓝色
            3: [1.0, 1.0, 0.0]  # 人行横道 - 黄色
        }
        H, W, C = data.shape
        # 创建RGB图像
        rgb_image = np.zeros((H, W, 3), dtype=np.float32)

        # 处理每个通道
        for c in range(C):
            channel_data = data[:, :, c]

            if is_binary:
                # 对于二值数据，将大于0.5的像素设为对应颜色
                mask = channel_data > 0.5
            else:
                # 对于连续数据，使用强度作为权重
                mask = channel_data > 0.1  # 使用较小的阈值
                intensity = channel_data

            if np.any(mask):
                color = np.array(channel_colors[c])

                if is_binary:
                    # 二值数据：直接使用颜色
                    for rgb_c in range(3):
                        rgb_image[mask, rgb_c] = np.maximum(rgb_image[mask, rgb_c], color[rgb_c])
                else:
                    # 连续数据：使用强度加权
                    for rgb_c in range(3):
                        weighted_color = color[rgb_c] * intensity[mask]
                        rgb_image[mask, rgb_c] = np.maximum(rgb_image[mask, rgb_c], weighted_color)

        return rgb_image

    def create_detailed_visualization(self, input_data: np.ndarray,
                                      reconstructed_data: np.ndarray,
                                      sample_id: int,
                                      save_path: Path):
        """
        创建详细的可视化图像
        
        Args:
            input_data: 输入数据 (H, W, 4)
            reconstructed_data: 重建数据 (H, W, 4)
            sample_id: 样本ID
            save_path: 保存路径
        """
        channel_names = ['ego_trajectory', 'divider', 'drivable_area', 'ped_crossing']
        fig = plt.figure(figsize=(20, 15))

        # 第一行：输入的各个通道
        for c in range(4):
            plt.subplot(3, 4, c + 1)
            plt.imshow(input_data[:, :, c], cmap='gray', vmin=0, vmax=1)
            plt.title(f'input - {channel_names[c]}')
            plt.axis('off')

        # 第二行：重建的各个通道
        for c in range(4):
            plt.subplot(3, 4, c + 5)
            plt.imshow(reconstructed_data[:, :, c], cmap='gray', vmin=0, vmax=1)
            plt.title(f'reconstructed - {channel_names[c]}')
            plt.axis('off')

        # 第三行：叠加可视化
        plt.subplot(3, 4, 9)
        input_overlay = self.create_channel_overlay_visualization(input_data, is_binary=True)
        plt.imshow(input_overlay)
        plt.title('input - 4 channels overlay')
        plt.axis('off')

        plt.subplot(3, 4, 10)
        # 将重建结果二值化后进行叠加
        reconstructed_binary = (reconstructed_data > 0.5).astype(np.float32)
        reconstructed_overlay = self.create_channel_overlay_visualization(reconstructed_binary, is_binary=True)
        plt.imshow(reconstructed_overlay)
        plt.title('reconstructed - 4 channels overlay')
        plt.axis('off')

        plt.subplot(3, 4, 11)
        # 连续值的重建结果叠加
        reconstructed_continuous_overlay = self.create_channel_overlay_visualization(reconstructed_data,
                                                                                     is_binary=False)
        plt.imshow(reconstructed_continuous_overlay)
        plt.title('reconstructed - 4 channels overlay(continue)')
        plt.axis('off')

        # 添加颜色图例
        plt.subplot(3, 4, 12)
        legend_colors = ['red', 'green', 'blue', 'yellow']
        legend_labels = channel_names

        # 创建颜色块
        for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
            plt.barh(i, 1, color=color, alpha=0.7)
            plt.text(1.1, i, label, va='center', fontsize=12)

        plt.xlim(0, 2)
        plt.ylim(-0.5, 3.5)
        plt.title('channel legend')
        plt.axis('off')

        plt.tight_layout()

        # 保存图像
        save_file = save_path / f'detailed_visualization_sample_{sample_id:04d}.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return save_file

    def create_comparison_grid(self, input_data_list: List[np.ndarray],
                               reconstructed_data_list: List[np.ndarray],
                               save_path: Path,
                               max_samples: int = 4):
        """
        创建多样本对比网格
        
        Args:
            input_data_list: 输入数据列表
            reconstructed_data_list: 重建数据列表
            save_path: 保存路径
            max_samples: 最大样本数
        """
        num_samples = min(len(input_data_list), max_samples)

        # 创建对比网格：每行显示一个样本的输入和重建叠加图
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 输入叠加
            input_overlay = self.create_channel_overlay_visualization(
                input_data_list[i], is_binary=True
            )
            axes[i, 0].imshow(input_overlay)
            axes[i, 0].set_title(f'scene {i + 1} input')
            axes[i, 0].axis('off')

            # 重建叠加（二值化）
            reconstructed_binary = (reconstructed_data_list[i] > 0.5).astype(np.float32)
            reconstructed_overlay = self.create_channel_overlay_visualization(
                reconstructed_binary, is_binary=True
            )
            axes[i, 1].imshow(reconstructed_overlay)
            axes[i, 1].set_title(f'scene {i + 1} reconstructed')
            axes[i, 1].axis('off')

        plt.tight_layout()

        # 保存对比网格
        save_file = save_path / 'comparison_grid.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return save_file


def main():
    """主函数"""
    # 配置文件路径
    config_path = '/config/autoencoder_config_v1.yaml'

    # 加载配置
    config = load_config(config_path)

    # 模型路径
    model_path = '/home/zhangxueyou/PycharmProjects/scene_reco2/models/autoencoder_best.pth'

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型")
        return

    # 创建推理器
    inference = AutoencoderInference(config, model_path)

    # 运行推理
    inference.run_inference()


if __name__ == "__main__":
    main()
