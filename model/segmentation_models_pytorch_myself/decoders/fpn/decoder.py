import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, upsample=True):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.upsample = upsample
        
    def forward(self, x, skip=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1], upsample=True)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2], upsample=True)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0] # [1, 1, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:] # 1/4, 1/8, 1/16, 1/32 if downsample8 1/4, 1/8, 1/8, 1/16

        # 通过1*1卷积 将特征映射到pyramid_channels 相邻级特征通过nearst插值到相同空间维度 然后相加
        p5 = self.p5(c5) # 1/4 32*32
        p4 = self.p4(p5, c4) # 1/4 64*64
        p3 = self.p3(p4, c3) # 64*64
        p2 = self.p2(p3, c2) # 128*128

        # 将所有级别特征上采样到1/4空间分辨率 如 1/32通过三次3*3卷积和bilinear插值实现
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        
        # to cal MAdds
        # feature_pyramid1 = self.seg_blocks[0](p5)
        # feature_pyramid2 = self.seg_blocks[1](p4)
        # feature_pyramid3 = self.seg_blocks[2](p3)
        # feature_pyramid4 = self.seg_blocks[3](p2)
        # feature_pyramid = [feature_pyramid1, feature_pyramid2, feature_pyramid3, feature_pyramid4]
        
        # 所有级别特征相加
        # x = sum(feature_pyramid)
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
    
if __name__ == '__main__':
    # (256, 128, 64, 32, 16)
    f1 = torch.rand([2, 32, 16, 16])
    f2 = torch.rand([2, 64, 8, 8])
    f3 = torch.rand([2, 128, 4, 4])
    f4 = torch.rand([2, 256, 2, 2])
    
    encoder_channels = [16, 32, 64, 128, 256]
    
    fpn_decoder = FPNDecoder(encoder_channels)
    
    output = fpn_decoder(f1, f2, f3, f4)
