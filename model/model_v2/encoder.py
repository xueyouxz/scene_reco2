from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块，提升特征复用"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)

        avg_out = self.fc(avg_y)
        max_out = self.fc(max_y)

        y = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv(x_cat)
        return x * torch.sigmoid(x_cat)


class ImprovedSceneEncoder(nn.Module):
    """改进的场景编码器，融合残差连接和注意力机制"""

    def __init__(self, config: Dict):
        super().__init__()
        encoder_config = config['model_config']['encoder']
        data_config = config['data_processing']

        # 动态计算输入参数
        self.use_ego_trajectory = data_config.get('use_ego_trajectory', True)
        self.base_channels = data_config.get('base_channels', 3)
        self.input_channels = self.base_channels + (1 if self.use_ego_trajectory else 0)
        
        # 使用新的栅格尺寸配置
        self.input_height = data_config.get('grid_height', data_config.get('input_height', 512))
        self.input_width = data_config.get('grid_width', data_config.get('input_width', 512))
        self.hidden_dim = encoder_config['hidden_dim']
        self.use_attention = encoder_config.get('use_attention', True)
        
        print(f"编码器配置:")
        print(f"  输入通道数: {self.input_channels} (基础: {self.base_channels}, 自车轨迹: {self.use_ego_trajectory})")
        print(f"  输入尺寸: {self.input_height}x{self.input_width}")
        print(f"  隐藏维度: {self.hidden_dim}")

        # 多尺度特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 残差编码块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 2)
        ])

        # 注意力模块
        if self.use_attention:
            self.channel_attention = nn.ModuleList([
                ChannelAttention(128),
                ChannelAttention(256),
                ChannelAttention(512),
                ChannelAttention(512)
            ])
            self.spatial_attention = SpatialAttention()

        # 计算特征图尺寸
        self.feature_shapes = []
        h, w = self.input_height, self.input_width
        h, w = h // 2, w // 2  # initial_conv
        for _ in range(4):
            h, w = h // 2, w // 2
            self.feature_shapes.append((h, w))

        self.final_height = h
        self.final_width = w
        self.final_channels = 512

        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pool = nn.AdaptiveAvgPool2d((2, 2))

        # 多层感知机编码到隐藏维度
        fc_input_dim = 512 * (1 + 4)  # global + local features
        self.fc_encoder = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # 保存中间特征用于跳跃连接
        self.skip_features = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        返回: (编码特征, 跳跃连接特征列表)
        """
        # 调整维度顺序
        x = x.permute(0, 3, 1, 2)

        # 初始特征提取
        x = self.initial_conv(x)

        skip_features = []

        # 通过残差块和注意力模块
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)

            if self.use_attention:
                x = self.channel_attention[i](x)
                if i == len(self.res_blocks) - 1:
                    x = self.spatial_attention(x)

            skip_features.append(x)

        # 全局和局部特征聚合
        global_feat = self.global_pool(x).flatten(1)
        local_feat = self.local_pool(x).flatten(1)

        # 组合特征
        combined_feat = torch.cat([global_feat, local_feat], dim=1)

        # 编码到隐藏维度
        encoded = self.fc_encoder(combined_feat)

        return encoded, skip_features

    def get_output_info(self) -> Dict:
        return {
            'final_channels': self.final_channels,
            'final_height': self.final_height,
            'final_width': self.final_width,
            'hidden_dim': self.hidden_dim,
            'feature_shapes': self.feature_shapes
        }