#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的训练脚本，包含早停、学习率调度、梯度裁剪等
"""

import json
import time
from pathlib import Path
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.model_v2.tools.data_loader import create_data_loaders, load_config
from model.model_v2.autoencoder import create_autoencoder


class ImprovedAutoEncoderTrainer:
    """改进的训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()

        # 创建目录
        self._create_directories()

        # 设置随机种子
        self._set_random_seed()

        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)

        # 创建模型
        self.model = create_autoencoder(config).to(self.device)

        # 创建优化器
        self.optimizer = self._create_optimizer()

        # 混合精度训练
        self.use_amp = config['training_config'].get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()

        # 学习率调度器
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler()

        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config['training_config'].get('early_stopping_patience', 10),
            min_delta=config['training_config'].get('early_stopping_delta', 1e-4)
        )

        # 最佳模型追踪
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _get_device(self) -> torch.device:
        """获取设备"""
        if self.config['device_config']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device_config']['gpu_id']}")
            print(f"使用GPU: {device}")
            # 设置cuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = torch.device('cpu')
            print("使用CPU")
        return device

    def _create_directories(self):
        """创建必要的目录"""
        dirs_to_create = [
            self.config['training_config']['checkpoint']['save_dir'],
            self.config['output_config']['model_save_path'],
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _set_random_seed(self):
        """设置随机种子"""
        seed = self.config['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['training_config']['optimizer_config']
        optimizer_type = optimizer_config['type'].lower()

        # 参数分组（编码器和解码器使用不同学习率）
        encoder_params = self.model.encoder.parameters()
        decoder_params = self.model.decoder.parameters()

        param_groups = [
            {'params': encoder_params, 'lr': optimizer_config['encoder_lr']},
            {'params': decoder_params, 'lr': optimizer_config['decoder_lr']}
        ]

        if optimizer_type == 'adamw':
            return optim.AdamW(
                param_groups,
                weight_decay=optimizer_config['weight_decay'],
                betas=(optimizer_config['beta1'], optimizer_config['beta2'])
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                param_groups,
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config['training_config']['scheduler_config']
        scheduler_type = scheduler_config['type'].lower()

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config['T_mult'],
                eta_min=scheduler_config['eta_min']
            )
        elif scheduler_type == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[scheduler_config['max_encoder_lr'],
                        scheduler_config['max_decoder_lr']],
                epochs=self.config['training_config']['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=scheduler_config['pct_start']
            )
        else:
            return None

    def _create_warmup_scheduler(self):
        """创建预热调度器"""
        warmup_config = self.config['training_config'].get('warmup_config', {})
        if warmup_config.get('enabled', False):
            return WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_config['epochs'],
                warmup_factor=warmup_config['factor']
            )
        return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'iou': 0.0
        }
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch + 1}")

        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 混合精度训练
            if self.use_amp:
                with autocast():
                    reconstructed, encoded = self.model(inputs)
                    losses = self.model.compute_loss(targets, reconstructed, encoded)
                    total_loss = losses['total_loss']

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training_config'].get('grad_clip', 1.0)
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, encoded = self.model(inputs)
                losses = self.model.compute_loss(targets, reconstructed, encoded)
                total_loss = losses['total_loss']

                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training_config'].get('grad_clip', 1.0)
                )

                self.optimizer.step()

            # 更新指标
            metrics['loss'] += total_loss.item()
            metrics['accuracy'] += losses['mean_accuracy'].item()
            metrics['iou'] += losses['iou'].item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{losses['mean_accuracy'].item():.3f}",
                'IoU': f"{losses['iou'].item():.3f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # 步级学习率调度
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

        # 计算平均指标
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        # 动态计算通道数
        data_config = self.config['data_processing']
        num_channels = data_config.get('base_channels', 3) + (1 if data_config.get('use_ego_trajectory', True) else 0)
        
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'iou': 0.0,
            'channel_acc': [0.0] * num_channels
        }
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets, metadata in tqdm(self.val_loader, desc="验证"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.use_amp:
                    with autocast():
                        reconstructed, encoded = self.model(inputs)
                        losses = self.model.compute_loss(targets, reconstructed, encoded)
                else:
                    reconstructed, encoded = self.model(inputs)
                    losses = self.model.compute_loss(targets, reconstructed, encoded)

                metrics['loss'] += losses['total_loss'].item()
                metrics['accuracy'] += losses['mean_accuracy'].item()
                metrics['iou'] += losses['iou'].item()

                # 通道级准确率
                for c in range(num_channels):
                    if f'channel_{c}_accuracy' in losses:
                        metrics['channel_acc'][c] += losses[f'channel_{c}_accuracy'].item()

        # 计算平均指标
        metrics['loss'] /= num_batches
        metrics['accuracy'] /= num_batches
        metrics['iou'] /= num_batches
        for c in range(num_channels):
            metrics['channel_acc'][c] /= num_batches

        return metrics

    def train(self):
        """完整训练流程"""
        epochs = self.config['training_config']['epochs']

        for epoch in range(epochs):
            start_time = time.time()

            # 预热学习率
            if self.warmup_scheduler:
                self.warmup_scheduler.step(epoch)
            # 训练
            train_metrics = self.train_epoch(epoch)
            # 验证
            val_metrics = self.validate_epoch(epoch)

            # 学习率调度
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            # 早停检查
            if self.early_stopping(val_metrics['loss']):
                print(f"早停触发，停止训练")
                break

            # 定期保存检查点
            if (epoch + 1) % self.config['training_config']['checkpoint']['save_freq'] == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        # 保存最终模型和历史
        self._save_final_model()

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
        }

        save_dir = Path(self.config['training_config']['checkpoint']['save_dir'])

        if is_best:
            save_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"最佳模型保存到: {save_path}")
        else:
            save_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, save_path)
            print(f"检查点保存到: {save_path}")

    def _save_final_model(self):
        """保存最终模型"""
        model_save_path = Path(self.config['output_config']['model_save_path'])

        # 保存最佳模型
        if self.best_model_state:
            best_model_path = model_save_path / 'autoencoder_best.pth'
            torch.save({
                'model_state_dict': self.best_model_state,
                'model_info': self.model.get_model_info(),
                'config': self.config
            }, best_model_path)
            print(f"最佳模型保存到: {best_model_path}")



class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class WarmupScheduler:
    """预热学习率调度器"""

    def __init__(self, optimizer, warmup_epochs: int, warmup_factor: float = 0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = self.warmup_factor + (1 - self.warmup_factor) * epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * factor


def main():
    """主函数"""
    config_path = '/config/autoencoder_config_v2_3channels.yaml'
    config = load_config(config_path)

    trainer = ImprovedAutoEncoderTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()