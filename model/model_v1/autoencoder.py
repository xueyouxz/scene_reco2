import os
import sys
from typing import Dict
import torch
import torch.nn as nn

# 添加模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder import SceneEncoder
from decoder import SceneDecoder


class SceneAutoencoder(nn.Module):
    """简化场景自编码器模型"""
    
    def __init__(self, config: Dict):
        """
        初始化自编码器
        
        Args:
            config: 模型配置字典
        """
        super(SceneAutoencoder, self).__init__()
        
        self.config = config
        
        # 创建编码器
        self.encoder = SceneEncoder(config)
        
        # 获取编码器输出信息
        encoder_output_info = self.encoder.get_output_info()
        
        # 创建解码器
        self.decoder = SceneDecoder(config, encoder_output_info)
        
        # 损失函数
        self.loss_function = self._get_loss_function()
        
    def _get_loss_function(self):
        """获取损失函数"""
        loss_type = self.config['training_config']['loss_function'].lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'bce':
            return nn.BCELoss()  # 二元交叉熵损失，适合二值数据
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (N, H, W, C) = (N, 700, 400, 4)
            
        Returns:
            重建图像 (N, H, W, C) = (N, 700, 400, 4)
        """
        # 编码
        encoded = self.encoder(x)  # (N, hidden_dim)
        
        # 解码
        reconstructed = self.decoder(encoded)  # (N, 700, 400, 4)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅编码，返回潜在编码
        
        Args:
            x: 输入张量 (N, H, W, C)
            
        Returns:
            潜在编码 (N, hidden_dim)
        """
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        仅解码，从潜在编码重建图像
        
        Args:
            encoded: 潜在编码 (N, hidden_dim)
            
        Returns:
            重建图像 (N, H, W, C)
        """
        with torch.no_grad():
            reconstructed = self.decoder(encoded)
        return reconstructed
    
    def compute_loss(self, input_batch: torch.Tensor, 
                     reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            input_batch: 原始输入 (N, H, W, C) - 二值数据 (0或1)
            reconstructed: 重建输出 (N, H, W, C) - sigmoid输出 (0到1之间)
            
        Returns:
            损失字典
        """
        # 确保输入是二值的 (0或1)
        input_binary = (input_batch > 0.5).float()
        
        # 计算重建损失
        reconstruction_loss = self.loss_function(reconstructed, input_binary)
        
        # 计算二值化准确率（额外指标）
        reconstructed_binary = (reconstructed > 0.5).float()
        accuracy = (reconstructed_binary == input_binary).float().mean()
        
        losses = {
            'reconstruction_loss': reconstruction_loss,
            'total_loss': reconstruction_loss,
            'binary_accuracy': accuracy
        }
        
        return losses
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'hidden_dimension': self.config['model_config']['encoder']['hidden_dim'],
            'input_shape': (
                self.config['data_processing']['input_height'],
                self.config['data_processing']['input_width'],
                self.config['data_processing']['input_channels']
            )
        }


def create_autoencoder(config: Dict) -> SceneAutoencoder:
    """
    创建简化自编码器模型
    
    Args:
        config: 配置字典
        
    Returns:
        简化自编码器模型实例
    """
    model = SceneAutoencoder(config)
    return model
