#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的推理测试脚本
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.model_v2.tools.data_loader import create_data_loaders, load_config
from model.model_v2.autoencoder import create_autoencoder


class ImprovedAutoencoderInference:
    """改进的自编码器推理器"""

    def __init__(self, config: Dict, model_path: str):
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

            # 创建子目录
            (Path(results_dir) / 'detailed').mkdir(exist_ok=True)
            (Path(results_dir) / 'comparisons').mkdir(exist_ok=True)
            (Path(results_dir) / 'metrics').mkdir(exist_ok=True)

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        print(f"加载模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 使用保存的配置创建模型
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config

        model = create_autoencoder(model_config)

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

    def evaluate_model(self) -> Dict:
        """评估模型性能"""
        print("开始模型评估...")

        # 动态计算通道数
        data_config = self.config['data_processing']
        num_channels = data_config.get('base_channels', 3) + (1 if data_config.get('use_ego_trajectory', True) else 0)

        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'iou': 0.0,
            'channel_accuracy': [0.0] * num_channels,
            'channel_iou': [0.0] * num_channels,
            'focal_bce': 0.0,
            'ssim': 0.0,
            'mse': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for inputs, targets, metadata in tqdm(self.test_loader, desc="评估"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                reconstructed, encoded = self.model(inputs)

                # 计算损失
                losses = self.model.compute_loss(targets, reconstructed, encoded)

                # 累积指标
                metrics['loss'] += losses['total_loss'].item()
                metrics['accuracy'] += losses['mean_accuracy'].item()
                metrics['iou'] += losses['iou'].item()

                # 分解损失
                metrics['focal_bce'] += losses.get('focal_bce', 0.0).item()
                metrics['ssim'] += losses.get('ssim', 0.0).item()
                metrics['mse'] += losses.get('mse', 0.0).item()

                # 通道级指标
                for c in range(num_channels):
                    if f'channel_{c}_accuracy' in losses:
                        metrics['channel_accuracy'][c] += losses[f'channel_{c}_accuracy'].item()

                    # 计算通道级IoU
                    rec_binary = (reconstructed[:, :, :, c] > 0.5).float()
                    tgt_binary = (targets[:, :, :, c] > 0.5).float()
                    intersection = (rec_binary * tgt_binary).sum()
                    union = ((rec_binary + tgt_binary) > 0).float().sum()
                    channel_iou = (intersection / (union + 1e-6)).item()
                    metrics['channel_iou'][c] += channel_iou

                num_batches += 1

        # 计算平均值
        for key in metrics:
            if isinstance(metrics[key], list):
                metrics[key] = [v / num_batches for v in metrics[key]]
            else:
                metrics[key] = metrics[key] / num_batches

        # 打印评估结果
        print("\n" + "=" * 50)
        print("模型评估结果:")
        print("=" * 50)
        print(f"总体损失: {metrics['loss']:.4f}")
        print(f"平均准确率: {metrics['accuracy']:.3f}")
        print(f"平均IoU: {metrics['iou']:.3f}")
        print("\n分解损失:")
        print(f"  Focal BCE: {metrics['focal_bce']:.4f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print("\n通道级性能:")
        # 动态获取通道名称
        channel_names = ['divider', 'drivable_area', 'ped_crossing']
        if data_config.get('use_ego_trajectory', True):
            channel_names = ['ego_trajectory'] + channel_names

        for c, name in enumerate(channel_names):
            if c < len(metrics['channel_accuracy']):
                print(f"  {name}:")
                print(f"    准确率: {metrics['channel_accuracy'][c]:.3f}")
                print(f"    IoU: {metrics['channel_iou'][c]:.3f}")

        # 保存评估结果
        results_dir = Path(self.config['visualization_config']['test_visualization']['results_dir'])
        metrics_file = results_dir / 'metrics' / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n评估结果保存到: {metrics_file}")

        return metrics

    def visualize_reconstruction_results(self, num_samples: int = None):
        """可视化重建结果"""
        if not self.config['visualization_config']['test_visualization']['enabled']:
            print("测试可视化未启用")
            return

        if num_samples is None:
            num_samples = self.config['visualization_config']['test_visualization']['num_test_samples']

        results_dir = Path(self.config['visualization_config']['test_visualization']['results_dir'])

        self.model.eval()
        sample_count = 0

        reconstruction_data = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, metadata) in enumerate(self.test_loader):
                if sample_count >= num_samples:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                reconstructed, encoded = self.model(inputs)

                # 计算损失
                losses = self.model.compute_loss(targets, reconstructed, encoded)

                # 转换为numpy
                inputs_np = inputs.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()

                batch_size = inputs_np.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    sample_data = {
                        'input': inputs_np[i],
                        'reconstructed': reconstructed_np[i],
                        'sample_id': sample_count,
                        'scene_token': metadata['scene_token'][
                            i] if 'scene_token' in metadata else f'sample_{sample_count}'
                    }
                    reconstruction_data.append(sample_data)

                    # 创建详细可视化
                    print(f"生成样本 {sample_count} 的详细可视化...")
                    self.create_enhanced_visualization(
                        inputs_np[i],
                        reconstructed_np[i],
                        sample_count,
                        results_dir / 'detailed',
                        scene_token=sample_data['scene_token']
                    )

                    sample_count += 1

        # 创建对比网格
        if reconstruction_data:
            print("生成对比网格...")
            self.create_comparison_grid_enhanced(
                reconstruction_data[:min(8, len(reconstruction_data))],
                results_dir / 'comparisons'
            )

        # 保存重建数据
        np.savez_compressed(
            results_dir / 'reconstruction_data.npz',
            **{f'sample_{i}': data for i, data in enumerate(reconstruction_data)}
        )
        print(f"重建数据已保存")

    def create_enhanced_visualization(self, input_data: np.ndarray,
                                      reconstructed_data: np.ndarray,
                                      sample_id: int,
                                      save_path: Path,
                                      scene_token: str = None):
        """创建增强的可视化"""
        # 动态获取通道名称
        data_config = self.config['data_processing']
        if data_config.get('use_ego_trajectory', True):
            channel_names = ['Ego Trajectory', 'Lane Divider', 'Drivable Area', 'Pedestrian Crossing']
            channel_colors = ['red', 'green', 'blue', 'yellow']
        else:
            channel_names = ['Lane Divider', 'Drivable Area', 'Pedestrian Crossing']
            channel_colors = ['green', 'blue', 'yellow']

        num_channels = len(channel_names)

        fig = plt.figure(figsize=(24, 16))

        # 设置样式
        # plt.style.use('seaborn-v0_8-darkgrid')

        # 创建子图布局
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

        # 计算指标
        reconstructed_binary = (reconstructed_data > 0.5).astype(np.float32)
        input_binary = (input_data > 0.5).astype(np.float32)

        # 第1-2行：各通道对比
        for c in range(num_channels):
            # 输入
            ax1 = fig.add_subplot(gs[c // 2, c % 2 * 2])
            im1 = ax1.imshow(input_data[:, :, c], cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f'Input - {channel_names[c]}', fontsize=10, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 重建
            ax2 = fig.add_subplot(gs[c // 2, c % 2 * 2 + 1])
            im2 = ax2.imshow(reconstructed_data[:, :, c], cmap='viridis', vmin=0, vmax=1)
            ax2.set_title(f'Reconstructed - {channel_names[c]}', fontsize=10, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 添加指标
            acc = np.mean(reconstructed_binary[:, :, c] == input_binary[:, :, c])
            intersection = np.sum(reconstructed_binary[:, :, c] * input_binary[:, :, c])
            union = np.sum((reconstructed_binary[:, :, c] + input_binary[:, :, c]) > 0)
            iou = intersection / (union + 1e-6)

            ax2.text(0.02, 0.98, f'Acc: {acc:.3f}\nIoU: {iou:.3f}',
                     transform=ax2.transAxes, fontsize=8,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                                                        facecolor='white', alpha=0.8))

        # 第3行：叠加可视化
        ax3 = fig.add_subplot(gs[2, :2])
        input_overlay = self.create_channel_overlay(input_data, is_binary=True)
        ax3.imshow(input_overlay)
        ax3.set_title('Input - All Channels Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[2, 2:4])
        reconstructed_overlay = self.create_channel_overlay(reconstructed_binary, is_binary=True)
        ax4.imshow(reconstructed_overlay)
        ax4.set_title('Reconstructed - All Channels Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # 第3行右侧：差异图
        ax5 = fig.add_subplot(gs[2, 4])
        diff = np.abs(input_binary - reconstructed_binary).mean(axis=2)
        im5 = ax5.imshow(diff, cmap='hot', vmin=0, vmax=1)
        ax5.set_title('Difference Map', fontsize=10, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # 第4行：指标可视化
        ax6 = fig.add_subplot(gs[3, :3])
        channel_acc = [np.mean(reconstructed_binary[:, :, c] == input_binary[:, :, c])
                       for c in range(num_channels)]
        channel_iou = []
        for c in range(num_channels):
            intersection = np.sum(reconstructed_binary[:, :, c] * input_binary[:, :, c])
            union = np.sum((reconstructed_binary[:, :, c] + input_binary[:, :, c]) > 0)
            channel_iou.append(intersection / (union + 1e-6))

        x = np.arange(len(channel_names))
        width = 0.35

        ax6.bar(x - width / 2, channel_acc, width, label='Accuracy', color='skyblue', edgecolor='navy')
        ax6.bar(x + width / 2, channel_iou, width, label='IoU', color='lightcoral', edgecolor='darkred')

        ax6.set_xlabel('Channel', fontweight='bold')
        ax6.set_ylabel('Score', fontweight='bold')
        ax6.set_title('Channel-wise Performance Metrics', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(channel_names, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])

        # 添加数值标签
        for i, (acc, iou) in enumerate(zip(channel_acc, channel_iou)):
            ax6.text(i - width / 2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            ax6.text(i + width / 2, iou + 0.01, f'{iou:.3f}', ha='center', va='bottom', fontsize=8)

        # 第4行右侧：图例
        ax7 = fig.add_subplot(gs[3, 3:])
        ax7.axis('off')

        # 创建彩色图例
        for i, (color, name) in enumerate(zip(channel_colors, channel_names)):
            rect = patches.Rectangle((0.1, 0.8 - i * 0.2), 0.15, 0.1,
                                     linewidth=1, edgecolor='black',
                                     facecolor=color, alpha=0.7)
            ax7.add_patch(rect)
            ax7.text(0.3, 0.85 - i * 0.2, name, fontsize=11,
                     verticalalignment='center', fontweight='bold')

        # 添加总体指标
        total_acc = np.mean(channel_acc)
        total_iou = np.mean(channel_iou)
        ax7.text(0.5, 0.3, f'Overall Performance:\nAccuracy: {total_acc:.3f}\nIoU: {total_iou:.3f}',
                 fontsize=12, fontweight='bold', bbox=dict(boxstyle='round',
                                                           facecolor='lightgray', alpha=0.8))

        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)

        # 添加标题
        if scene_token:
            fig.suptitle(f'Scene Reconstruction Analysis - {scene_token}',
                         fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Scene Reconstruction Analysis - Sample {sample_id}',
                         fontsize=16, fontweight='bold')

        # 保存图像
        save_file = save_path / f'enhanced_visualization_sample_{sample_id:04d}.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return save_file

    def create_channel_overlay(self, data: np.ndarray, is_binary: bool = True) -> np.ndarray:
        """创建通道叠加可视化"""
        # 动态调整颜色映射
        data_config = self.config['data_processing']
        if data_config.get('use_ego_trajectory', True):
            channel_colors = {
                0: [1.0, 0.0, 0.0],  # 红 - ego_trajectory
                1: [0.0, 1.0, 0.0],  # 绿 - divider
                2: [0.0, 0.0, 1.0],  # 蓝 - drivable_area
                3: [1.0, 1.0, 0.0]  # 黄 - ped_crossing
            }
        else:
            channel_colors = {
                0: [0.0, 1.0, 0.0],  # 绿 - divider
                1: [0.0, 0.0, 1.0],  # 蓝 - drivable_area
                2: [1.0, 1.0, 0.0]  # 黄 - ped_crossing
            }

        H, W, C = data.shape
        rgb_image = np.zeros((H, W, 3), dtype=np.float32)

        for c in range(C):
            channel_data = data[:, :, c]

            if is_binary:
                mask = channel_data > 0.5
            else:
                mask = channel_data > 0.1
                intensity = channel_data

            if np.any(mask):
                color = np.array(channel_colors[c])

                if is_binary:
                    for rgb_c in range(3):
                        rgb_image[mask, rgb_c] = np.maximum(rgb_image[mask, rgb_c],
                                                            color[rgb_c] * 0.7)
                else:
                    for rgb_c in range(3):
                        weighted_color = color[rgb_c] * intensity[mask]
                        rgb_image[mask, rgb_c] = np.maximum(rgb_image[mask, rgb_c],
                                                            weighted_color)

        return rgb_image

    def create_comparison_grid_enhanced(self, reconstruction_data: List[Dict],
                                        save_path: Path):
        """创建增强的对比网格"""
        num_samples = len(reconstruction_data)

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, data in enumerate(reconstruction_data):
            input_data = data['input']
            reconstructed_data = data['reconstructed']

            # 输入叠加
            input_overlay = self.create_channel_overlay(input_data, is_binary=True)
            axes[i, 0].imshow(input_overlay)
            axes[i, 0].set_title(f'Input - {data["scene_token"][:20]}...', fontsize=10)
            axes[i, 0].axis('off')

            # 重建叠加
            reconstructed_binary = (reconstructed_data > 0.5).astype(np.float32)
            reconstructed_overlay = self.create_channel_overlay(reconstructed_binary, is_binary=True)
            axes[i, 1].imshow(reconstructed_overlay)
            axes[i, 1].set_title('Reconstructed', fontsize=10)
            axes[i, 1].axis('off')

            # 差异图
            diff = np.abs((input_data > 0.5).astype(np.float32) - reconstructed_binary).mean(axis=2)
            im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title('Difference', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.suptitle('Multi-Sample Reconstruction Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_file = save_path / 'enhanced_comparison_grid.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return save_file

    def run_inference(self):
        """运行完整推理流程"""
        # 评估模型
        metrics = self.evaluate_model()

        # 可视化结果
        self.visualize_reconstruction_results()

        print("\n推理完成！")


def main():
    """主函数"""
    config_path = Path(__file__).resolve().parents[3] / 'config' / 'autoencoder_config_v2_3channels.yaml'

    # 加载配置
    config = load_config(config_path)

    # 模型路径
    model_path = '/home/zhangxueyou/PycharmProjects/scene_reco2/ckpt/checkpoints_3ch/best_model.pth'

    inference = ImprovedAutoencoderInference(config, model_path)

    # 运行推理
    inference.run_inference()


if __name__ == "__main__":
    main()
