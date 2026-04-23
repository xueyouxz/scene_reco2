from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleBlock(nn.Module):
    """改进的上采样块，避免棋盘效应"""

    def __init__(self, in_channels: int, out_channels: int,
                 skip_channels: int = 0, use_attention: bool = True):
        super().__init__()

        total_channels = in_channels + skip_channels

        # 使用双线性插值 + 卷积替代转置卷积
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv_block = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

    def forward(self, x, skip_feat=None):
        x = self.upsample(x)

        if skip_feat is not None:
            # 调整skip feature的尺寸
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:],
                                          mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)

        x = self.conv_block(x)

        if self.attention is not None:
            att_map = self.attention(x)
            x = x * att_map

        return x


class ImprovedSceneDecoder(nn.Module):
    """改进的解码器，支持跳跃连接和多尺度特征融合"""

    def __init__(self, config: Dict, encoder_output_info: Dict):
        super().__init__()

        decoder_config = config['model_config']['decoder']
        data_config = config['data_processing']

        # 动态计算输出参数
        self.use_ego_trajectory = data_config.get('use_ego_trajectory', True)
        self.base_channels = data_config.get('base_channels', 3)
        self.output_channels = self.base_channels + (1 if self.use_ego_trajectory else 0)
        
        # 使用新的栅格尺寸配置
        self.output_height = data_config.get('grid_height', data_config.get('input_height', 512))
        self.output_width = data_config.get('grid_width', data_config.get('input_width', 512))
        self.hidden_dim = encoder_output_info['hidden_dim']
        self.use_skip_connections = decoder_config.get('use_skip_connections', True)
        
        print(f"解码器配置:")
        print(f"  输出通道数: {self.output_channels} (基础: {self.base_channels}, 自车轨迹: {self.use_ego_trajectory})")
        print(f"  输出尺寸: {self.output_height}x{self.output_width}")
        print(f"  隐藏维度: {self.hidden_dim}")

        # 从隐藏维度恢复到特征图
        # 初始特征图大小基于编码器的最终输出
        initial_size = 4  # 可以根据编码器调整
        initial_channels = 512

        self.fc_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, initial_channels * initial_size * initial_size),
            nn.ReLU(inplace=True)
        )

        self.initial_size = initial_size
        self.initial_channels = initial_channels

        self.initial_conv = nn.Sequential(
            nn.Conv2d(initial_channels, initial_channels, 3, 1, 1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )

        # 定义每个阶段的输出通道数
        channel_sequence = [512, 256, 128, 64]
        
        # 编码器的skip features按深度排序为 [128, 256, 512, 512]
        # 反转后为 [512, 512, 256, 128]
        skip_channel_sequence = [512, 512, 256, 128]

        # 上采样块（支持跳跃连接）
        self.up_blocks = nn.ModuleList()
        for i in range(len(channel_sequence) - 1):
            in_ch = channel_sequence[i]
            out_ch = channel_sequence[i + 1]

            if self.use_skip_connections:
                # 使用正确的跳跃连接通道数
                skip_ch = skip_channel_sequence[i]
                self.up_blocks.append(UpSampleBlock(in_ch, out_ch, skip_ch))
            else:
                self.up_blocks.append(UpSampleBlock(in_ch, out_ch, 0))

        # 添加最后一个上采样块从64到32
        final_channels = 32
        if self.use_skip_connections:
            # 最后一个skip connection来自编码器的第一层(128通道)
            self.up_blocks.append(UpSampleBlock(64, final_channels, 128))
        else:
            self.up_blocks.append(UpSampleBlock(64, final_channels, 0))

        # 多尺度特征融合 - 根据实际输出通道数配置
        # 收集来自不同层的特征：256, 128, 64, 32
        self.multi_scale_fusion = nn.ModuleList([
            nn.Conv2d(256, 32, 3, 1, 1),  # 来自up_blocks[0]的输出
            nn.Conv2d(128, 32, 3, 1, 1),  # 来自up_blocks[1]的输出
            nn.Conv2d(64, 32, 3, 1, 1),   # 来自up_blocks[2]的输出
            nn.Conv2d(32, 32, 3, 1, 1),   # 来自up_blocks[3]的输出
        ])

        # 最终输出层（通道独立处理）
        # 输入通道数 = final_channels + (多尺度特征数 * 32)
        fusion_channels = len(self.multi_scale_fusion) * 32
        total_final_channels = final_channels + fusion_channels

        self.channel_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_final_channels, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            ) for _ in range(self.output_channels)
        ])

        print(f"解码器初始化完成:")
        print(f"  初始特征图: {initial_channels}x{initial_size}x{initial_size}")
        print(f"  上采样块数量: {len(self.up_blocks)}")
        print(f"  多尺度融合模块数量: {len(self.multi_scale_fusion)}")
        print(f"  最终通道数: {total_final_channels}")

    def forward(self, encoded: torch.Tensor,
                skip_features: List[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            encoded: 编码特征 (N, hidden_dim)
            skip_features: 跳跃连接特征列表
        """
        # 全连接解码
        x = self.fc_decoder(encoded)
        x = x.reshape(x.size(0), self.initial_channels, self.initial_size, self.initial_size)
        x = self.initial_conv(x)

        # 反转跳跃连接特征顺序（从深到浅）
        if skip_features is not None and self.use_skip_connections:
            skip_features = skip_features[::-1]

        # 存储多尺度特征
        multi_scale_features = []
        feature_outputs = []  # 存储每层的输出用于多尺度融合

        # 逐层上采样
        for i, up_block in enumerate(self.up_blocks):
            if self.use_skip_connections and skip_features is not None and i < len(skip_features):
                x = up_block(x, skip_features[i])
            else:
                x = up_block(x)

            # 保存每层的输出
            feature_outputs.append(x)

        # 收集多尺度特征（从所有上采样块）
        # up_blocks[0]: 512+512->256通道 -> multi_scale_fusion[0]
        # up_blocks[1]: 256+512->128通道 -> multi_scale_fusion[1]
        # up_blocks[2]: 128+256->64通道 -> multi_scale_fusion[2]
        # up_blocks[3]: 64+128->32通道 -> multi_scale_fusion[3]
        for i in range(min(len(feature_outputs), len(self.multi_scale_fusion))):
            feat = feature_outputs[i]
            # 应用对应的融合模块
            fused_feat = self.multi_scale_fusion[i](feat)
            multi_scale_features.append(fused_feat)

        # 取最后一层的输出作为主特征
        x = feature_outputs[-1]

        # 调整所有特征到相同尺寸
        target_size = x.shape[2:]
        aligned_features = []
        for feat in multi_scale_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # 特征拼接
        if aligned_features:
            x = torch.cat([x] + aligned_features, dim=1)

        # 最终上采样到目标尺寸
        if x.shape[2] != self.output_height or x.shape[3] != self.output_width:
            x = F.interpolate(x, size=(self.output_height, self.output_width),
                              mode='bilinear', align_corners=False)

        # 通道独立解码
        channel_outputs = []
        for i, decoder in enumerate(self.channel_decoders):
            channel_out = decoder(x)
            channel_outputs.append(channel_out)

        # 组合所有通道
        output = torch.cat(channel_outputs, dim=1)

        # 调整维度顺序
        output = output.permute(0, 2, 3, 1)

        return output