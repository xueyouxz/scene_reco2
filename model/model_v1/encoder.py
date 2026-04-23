from typing import Dict, Tuple

import torch
import torch.nn as nn


class SceneEncoder(nn.Module):
    def __init__(self, config: Dict):
        """
        初始化编码器
        
        Args:
            config: 模型配置字典
        """
        super(SceneEncoder, self).__init__()
        
        # 从配置中提取参数
        encoder_config = config['model_config']['encoder']
        data_config = config['data_processing']
        
        self.input_channels = data_config['input_channels']  # 4
        self.input_height = data_config['input_height']      # 700
        self.input_width = data_config['input_width']        # 400
        self.hidden_dim = encoder_config['hidden_dim']       # 512
        
        self.conv_layers = encoder_config['conv_layers']     # 4
        self.base_channels = encoder_config['base_channels'] # 32
        self.channel_multiplier = encoder_config['channel_multiplier']  # 2
        self.kernel_size = encoder_config['kernel_size']     # 4
        self.stride = encoder_config['stride']               # 2
        self.padding = encoder_config['padding']             # 1
        self.use_batch_norm = encoder_config['use_batch_norm']
        self.activation = encoder_config['activation']
        
        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        
        current_channels = self.input_channels  # 4
        
        for i in range(self.conv_layers):
            out_channels = self.base_channels * (self.channel_multiplier ** i)
            
            # 卷积层
            conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            
            # 构建块：Conv + BN + Activation
            block = [conv]
            
            if self.use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            
            if self.activation == "relu":
                block.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                block.append(nn.LeakyReLU(0.2, inplace=True))
            
            self.conv_blocks.append(nn.Sequential(*block))
            current_channels = out_channels
        
        # 计算卷积后的特征图尺寸
        self.final_channels = current_channels
        self.final_height, self.final_width = self._calculate_output_size()
        
        # 全连接层到隐藏维度
        self.fc = nn.Linear(
            self.final_channels * self.final_height * self.final_width,
            self.hidden_dim
        )
        
        print(f"编码器最终特征图: {self.final_channels}x{self.final_height}x{self.final_width}")
        print(f"展平后维度: {self.final_channels * self.final_height * self.final_width}")
        print(f"隐藏层维度: {self.hidden_dim}")
        
    def _calculate_output_size(self) -> Tuple[int, int]:
        """计算卷积后的输出尺寸"""
        h, w = self.input_height, self.input_width
        
        for _ in range(self.conv_layers):
            h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
            w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return h, w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (N, H, W, C) = (N, 700, 400, 4)
            
        Returns:
            编码后的特征向量 (N, hidden_dim)
        """
        # 调整维度顺序：(N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (N, 4, 700, 400)
        
        # 通过卷积层
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # 展平
        x = x.reshape(x.size(0), -1)  # (N, final_channels * final_height * final_width)
        
        # 全连接层
        x = self.fc(x)  # (N, hidden_dim)
        
        return x
    
    def get_output_info(self) -> Dict:
        """获取编码器输出信息"""
        return {
            'final_channels': self.final_channels,
            'final_height': self.final_height,
            'final_width': self.final_width,
            'hidden_dim': self.hidden_dim
        }

