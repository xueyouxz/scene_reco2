import os
import sys
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder import ImprovedSceneEncoder
from decoder import ImprovedSceneDecoder


class FocalBCELoss(nn.Module):
    """Focal BCE Loss用于处理类别不平衡"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce_loss).mean()


class StructuralSimilarityLoss(nn.Module):
    """结构相似性损失"""

    def __init__(self, window_size=11, channel_weights=None):
        super().__init__()
        self.window_size = window_size
        self.channel_weights = channel_weights

    def gaussian_window(self, window_size, sigma=1.5):
        gauss = torch.tensor([torch.exp(-(x - window_size // 2) ** 2 /  torch.tensor(2 * sigma ** 2, dtype=torch.float32))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        # 处理每个通道
        _, _, _, C = pred.shape
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        ssim_loss = 0
        for c in range(C):
            window = self.create_window(self.window_size, 1).to(pred.device)

            mu1 = F.conv2d(pred[:, c:c + 1], window, padding=self.window_size // 2)
            mu2 = F.conv2d(target[:, c:c + 1], window, padding=self.window_size // 2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(pred[:, c:c + 1] * pred[:, c:c + 1], window,
                                 padding=self.window_size // 2) - mu1_sq
            sigma2_sq = F.conv2d(target[:, c:c + 1] * target[:, c:c + 1], window,
                                 padding=self.window_size // 2) - mu2_sq
            sigma12 = F.conv2d(pred[:, c:c + 1] * target[:, c:c + 1], window,
                               padding=self.window_size // 2) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            # weight = self.channel_weights[c] if self.channel_weights else 1.0
            weight = self.channel_weights[c]

            ssim_loss += weight * (1 - ssim_map.mean())

        return ssim_loss / C


class ImprovedSceneAutoencoder(nn.Module):
    """改进的场景自编码器"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 获取动态配置
        data_config = config['data_processing']
        self.use_ego_trajectory = data_config.get('use_ego_trajectory', True)
        self.base_channels = data_config.get('base_channels', 3)
        self.num_channels = self.base_channels + (1 if self.use_ego_trajectory else 0)

        # 创建改进的编码器和解码器
        self.encoder = ImprovedSceneEncoder(config)
        encoder_output_info = self.encoder.get_output_info()
        self.decoder = ImprovedSceneDecoder(config, encoder_output_info)

        # 多损失函数组合
        loss_config = config['training_config'].get('loss_config', {})

        # 动态调整通道权重
        default_weights = self._get_default_channel_weights()
        self.channel_weights = torch.tensor(
            loss_config.get('channel_weights', default_weights)
        )
        
        # 确保权重数量与通道数匹配
        if len(self.channel_weights) != self.num_channels:
            print(f"警告：通道权重数量({len(self.channel_weights)})与通道数({self.num_channels})不匹配，使用均匀权重")
            self.channel_weights = torch.ones(self.num_channels) / self.num_channels
        
        print(f"自编码器配置:")
        print(f"  通道数: {self.num_channels}")
        print(f"  通道权重: {self.channel_weights.tolist()}")

        # 损失函数组件
        self.focal_bce = FocalBCELoss(
            alpha=loss_config.get('focal_alpha', 0.25),
            gamma=loss_config.get('focal_gamma', 2.0)
        )
        self.ssim_loss = StructuralSimilarityLoss(
            channel_weights=self.channel_weights
        )
        self.mse_loss = nn.MSELoss()

        # 损失权重
        self.loss_weights = {
            'focal_bce': loss_config.get('focal_weight', 0.4),
            'ssim': loss_config.get('ssim_weight', 0.3),
            'mse': loss_config.get('mse_weight', 0.2),
            'regularization': loss_config.get('reg_weight', 0.1)
        }

    def _get_default_channel_weights(self) -> List[float]:
        """根据通道配置生成默认权重"""
        if self.use_ego_trajectory:
            # 4通道：[ego_trajectory, divider, drivable_area, ped_crossing]
            return [0.4, 0.25, 0.25, 0.1]  # 自车轨迹权重最高
        else:
            # 3通道：[divider, drivable_area, ped_crossing]
            return [0.4, 0.4, 0.2]  # 分隔线和可行驶区域权重相等

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        返回: (重建图像, 潜在编码)
        """
        # 编码
        encoded, skip_features = self.encoder(x)

        # 解码
        reconstructed = self.decoder(encoded, skip_features)

        return reconstructed, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码"""
        with torch.no_grad():
            encoded, _ = self.encoder(x)
        return encoded

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """仅解码"""
        with torch.no_grad():
            reconstructed = self.decoder(encoded, None)
        return reconstructed

    def compute_loss(self, input_batch: torch.Tensor,
                     reconstructed: torch.Tensor,
                     encoded: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        """
        # 确保输入是二值的
        input_binary = (input_batch > 0.5).float()

        losses = {}

        # 1. Focal BCE损失（处理类别不平衡）
        losses['focal_bce'] = self.focal_bce(reconstructed, input_binary)

        # 2. SSIM损失（保持结构信息）
        losses['ssim'] = self.ssim_loss(reconstructed, input_binary)

        # 3. MSE损失（像素级重建）
        losses['mse'] = self.mse_loss(reconstructed, input_binary)

        # 4. 正则化损失（约束潜在空间）
        if encoded is not None:
            losses['regularization'] = 0.01 * torch.mean(encoded ** 2)
        else:
            losses['regularization'] = torch.tensor(0.0).to(reconstructed.device)

        # 计算加权总损失
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
        losses['total_loss'] = total_loss

        # 计算通道级别的准确率
        reconstructed_binary = (reconstructed > 0.5).float()
        channel_accuracy = []
        for c in range(self.num_channels):
            acc = (reconstructed_binary[:, :, :, c] == input_binary[:, :, :, c]).float().mean()
            channel_accuracy.append(acc)
            losses[f'channel_{c}_accuracy'] = acc

        losses['mean_accuracy'] = sum(channel_accuracy) / len(channel_accuracy)

        # 计算IoU（交并比）
        intersection = (reconstructed_binary * input_binary).sum(dim=(1, 2, 3))
        union = ((reconstructed_binary + input_binary) > 0).float().sum(dim=(1, 2, 3))
        iou = (intersection / (union + 1e-6)).mean()
        losses['iou'] = iou

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
            'input_channels': self.num_channels,
            'use_ego_trajectory': self.use_ego_trajectory,
            'base_channels': self.base_channels,
            'grid_size': f"{self.config['data_processing'].get('grid_height', 512)}x{self.config['data_processing'].get('grid_width', 512)}",
            'architecture': 'Improved U-Net with Attention',
            'loss_functions': list(self.loss_weights.keys()),
            'channel_weights': self.channel_weights.tolist()
        }


def create_autoencoder(config: Dict) -> ImprovedSceneAutoencoder:
    """创建改进的自编码器模型"""
    return ImprovedSceneAutoencoder(config)