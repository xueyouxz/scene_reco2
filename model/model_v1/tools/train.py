#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化训练脚本
实现简化自编码器模型的训练过程，只保存最终权重
"""

from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm


from model.model_v1.tools.data_loader import create_data_loaders, load_config
from model.model_v1.autoencoder import create_autoencoder


class AutoencoderTrainer:

    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = self._get_device()
        
        # 创建必要的目录
        self._create_directories()
        
        # 设置随机种子
        self._set_random_seed()
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # 创建模型
        self.model = create_autoencoder(config).to(self.device)
        # 创建优化器和学习率调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        # 最佳模型状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
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
        dirs_to_create = [
            self.config['training_config']['checkpoint']['save_dir'],
            self.config['output_config']['model_save_path'],
            self.config['output_config']['logs_dir']
        ]
        
        if self.config['visualization_config']['test_visualization']['enabled']:
            dirs_to_create.append(
                self.config['visualization_config']['test_visualization']['results_dir']
            )
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _set_random_seed(self):
        """设置随机种子"""
        seed = self.config['random_seed']
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config['training_config']['optimizer'].lower()
        lr = self.config['training_config']['learning_rate']
        weight_decay = self.config['training_config']['weight_decay']
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config['training_config']['lr_scheduler']
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training_config']['epochs']
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            reconstructed = self.model(inputs)
            
            # 计算损失
            losses = self.model.compute_loss(targets, reconstructed)
            total_loss_batch = losses['total_loss']
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss_batch.item():.6f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.6f}",
                'Acc': f"{losses.get('binary_accuracy', 0.0):.3f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="验证"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                reconstructed = self.model(inputs)
                
                # 计算损失
                losses = self.model.compute_loss(targets, reconstructed)
                total_loss += losses['total_loss'].item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        epochs = self.config['training_config']['epochs']
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate_epoch(epoch)
            
            # 更新学习率调度器
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

        # 训练完成，保存最终模型
        self._save_final_model()
    
    def _save_final_model(self):
        """保存最终模型"""
        model_save_path = Path(self.config['output_config']['model_save_path'])
        
        # 保存最佳模型
        if self.best_model_state:
            best_model_path = model_save_path / 'autoencoder_best.pth'
            torch.save({
                'model_state_dict': self.best_model_state,
                'model_info': self.model.get_model_info()
            }, best_model_path)
            print(f"最佳模型保存到: {best_model_path}")

def main():
    """主函数"""
    config_path = Path(__file__).resolve().parents[3] / 'config' / 'base_autoencoder_v1.yaml'
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建训练器并开始训练
    trainer = AutoencoderTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
