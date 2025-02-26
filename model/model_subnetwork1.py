import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from .networks import get_norm_layer, SEBlock, DualDomainPriorFusionBlock, UnetBackbone


class DefaultModel(BaseModel):
    # prior_i & prior_p & m_shadow --> m
    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # for prior_i and prior_p
        self.ddpf = DualDomainPriorFusionBlock(init_dim, norm_layer, use_bias)

        # for m_shadow
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(init_dim, init_dim // 2)
        )
        self.backbone = UnetBackbone(init_dim, output_nc=init_dim, n_downsampling=4, use_conv_to_downsample=True,
                                     norm_type=norm_type, use_dropout=use_dropout, mode='res-bottleneck')
        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, 3, kernel_size=1, stride=1, bias=use_bias),
            nn.PReLU()
        )

    def forward(self, prior_i, prior_p, m_shadow):
        prior = self.ddpf(prior_i, prior_p)
        feature = self.feature_extraction(m_shadow)
        fused_feature = self.fusion(torch.cat([prior, feature], dim=1))
        backbone_out = self.backbone(fused_feature)
        m = m_shadow + self.out_block(backbone_out)
        m = torch.clamp(m, min=0, max=2 ** 0.5)
        return m
