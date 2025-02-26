import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from .networks import get_norm_layer, SEBlock, DualDomainPriorFusionBlock, TextureGuidedDemodulationBlock


class DefaultModel(BaseModel):
    # prior_i & prior_p & m, I_shadow --> I
    def __init__(self, init_dim=32, n_blocks=6, norm_type='instance'):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # for prior_i and prior_p
        self.ddpf = DualDomainPriorFusionBlock(init_dim, norm_layer, use_bias)

        # for m
        self.m_feature_extraction = nn.Sequential(
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
        self.m_feature_downsampling1 = nn.Sequential(
            nn.Conv2d(init_dim, init_dim * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.m_feature_downsampling2 = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # for I_shadow
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

        self.downsampling1 = nn.Sequential(
            nn.Conv2d(init_dim, init_dim * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.embed_blocks = nn.ModuleList(
            n_blocks * [TextureGuidedDemodulationBlock(init_dim * 4, init_dim * 4, norm_layer, use_bias)])
        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(init_dim * 4, init_dim * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(init_dim * 2, init_dim, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.multiplier_block = nn.Sequential(
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1, bias=use_bias),
            nn.ReLU(inplace=True)
        )
        self.bias_block = nn.Sequential(
            nn.Conv2d(init_dim, 3, kernel_size=1, stride=1, bias=use_bias),
            nn.PReLU()
        )

    def forward(self, prior_i, prior_p, m, I_shadow):
        prior = self.ddpf(prior_i, prior_p)

        feature_m = self.m_feature_extraction(m)
        feature_m = self.m_feature_downsampling1(feature_m)
        feature_m = self.m_feature_downsampling2(feature_m)

        feature = self.feature_extraction(I_shadow)
        fused_feature = self.fusion(torch.cat([prior, feature], dim=1))
        fused_feature = self.downsampling1(fused_feature)
        fused_feature = self.downsampling2(fused_feature)
        for embed_block in self.embed_blocks:
            fused_feature = embed_block(fused_feature, feature_m)
        fused_feature = self.upsampling1(fused_feature)
        fused_feature = self.upsampling2(fused_feature)

        multiplier = self.multiplier_block(fused_feature)
        bias = self.bias_block(fused_feature)
        I = I_shadow * multiplier + bias
        I = torch.clamp(I, min=0, max=2)

        return I
