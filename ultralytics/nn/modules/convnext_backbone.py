# convnext_backbone.py
import torch.nn as nn

from .convnext import ConvNeXt


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone for YOLO detection."""

    def __init__(self, model_size="tiny", out_indices=(1, 2, 3), pretrained=False):
        super().__init__()

        # 定义不同尺寸的ConvNeXt配置
        configs = {
            "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
            "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
            "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        }

        config = configs[model_size]
        self.out_indices = out_indices

        # 初始化ConvNeXt网络（去掉分类头）
        self.convnext = ConvNeXt(
            depths=config["depths"],
            dims=config["dims"],
            num_classes=0,  # 不需要分类头
        )

        # 移除最后的全局平均池化和分类头
        del self.convnext.norm
        del self.convnext.head

        self.out_channels = [config["dims"][i] for i in out_indices]

    def forward(self, x):
        features = []

        # 前向传播并收集多尺度特征
        for i in range(4):
            x = self.convnext.downsample_layers[i](x)
            x = self.convnext.stages[i](x)

            if i in self.out_indices:
                features.append(x)

        return features
