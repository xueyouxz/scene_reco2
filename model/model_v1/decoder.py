from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneDecoder(nn.Module):

    
    def __init__(self, config: Dict, encoder_output_info: Dict):
        """
        初始化解码器
        
        Args:
            config: 模型配置字典
            encoder_output_info: 编码器输出信息
        """
        super(SceneDecoder, self).__init__()
        
        # 从配置中提取参数
        decoder_config = config['model_config']['decoder']
        data_config = config['data_processing']
        encoder_config = config['model_config']['encoder']
        
        self.output_channels = data_config['input_channels']  # 4
        self.output_height = data_config['input_height']      # 700
        self.output_width = data_config['input_width']        # 400
        self.hidden_dim = encoder_config['hidden_dim']        # 512
        
        self.conv_layers = decoder_config['conv_layers']      # 4
        self.kernel_size = decoder_config['kernel_size']      # 4
        self.stride = decoder_config['stride']                # 2
        self.padding = decoder_config['padding']              # 1
        self.output_padding = decoder_config['output_padding'] # 0
        self.use_batch_norm = decoder_config['use_batch_norm']
        self.activation = decoder_config['activation']
        self.output_activation = decoder_config['output_activation']
        
        # 编码器输出信息
        self.encoder_final_channels = encoder_output_info['final_channels']
        self.encoder_final_height = encoder_output_info['final_height']
        self.encoder_final_width = encoder_output_info['final_width']
        
        # 全连接层：从隐藏维度恢复到特征图
        self.fc = nn.Linear(
            self.hidden_dim,
            self.encoder_final_channels * self.encoder_final_height * self.encoder_final_width
        )
        
        # 构建反卷积层
        self.deconv_blocks = nn.ModuleList()
        
        # 计算每层的通道数（逐层减少）
        channel_dims = [self.encoder_final_channels]
        for i in range(self.conv_layers - 1):
            channel_dims.append(channel_dims[-1] // 2)
        channel_dims[-1] = self.output_channels  # 最后一层输出原始通道数
        
        for i in range(self.conv_layers):
            in_channels = channel_dims[i]
            out_channels = channel_dims[i + 1] if i < self.conv_layers - 1 else self.output_channels
            
            # 反卷积层
            deconv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding
            )
            
            # 构建块：ConvTranspose + BN + Activation
            block = [deconv]
            
            # 最后一层不使用批标准化
            if self.use_batch_norm and i < self.conv_layers - 1:
                block.append(nn.BatchNorm2d(out_channels))
            
            # 激活函数
            if i < self.conv_layers - 1:
                if self.activation == "relu":
                    block.append(nn.ReLU(inplace=True))
                elif self.activation == "leaky_relu":
                    block.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                # 最后一层使用输出激活函数
                if self.output_activation == "sigmoid":
                    block.append(nn.Sigmoid())
                elif self.output_activation == "tanh":
                    block.append(nn.Tanh())
            
            self.deconv_blocks.append(nn.Sequential(*block))
        
        print(f"解码器通道维度: {channel_dims}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 编码后的特征向量 (N, hidden_dim)
            
        Returns:
            重建的图像 (N, H, W, C) = (N, 700, 400, 4)
        """
        # 全连接层
        x = self.fc(x)  # (N, encoder_final_channels * encoder_final_height * encoder_final_width)
        
        # 重塑为特征图
        x = x.reshape(
            x.size(0), 
            self.encoder_final_channels,
            self.encoder_final_height,
            self.encoder_final_width
        )  # (N, C, H, W)
        
        # 通过反卷积层
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
        
        # 最终尺寸调整（如果需要）
        if x.size(2) != self.output_height or x.size(3) != self.output_width:
            x = F.interpolate(
                x, 
                size=(self.output_height, self.output_width),
                mode='bilinear', 
                align_corners=False
            )
        
        # 调整维度顺序：(N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (N, 700, 400, 4)
        
        return x

