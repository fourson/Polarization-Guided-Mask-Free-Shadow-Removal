import functools

import torch
import torch.nn as nn
from torchvision import models

VGG19_FEATURES = models.vgg19(pretrained=True).features
CONV3_3_IN_VGG_19 = VGG19_FEATURES[0:15].cuda()

import warnings

warnings.filterwarnings("error")


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class GlobalAvgPool(nn.Module):
    """(N,C,H,W) -> (N,C)"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, -1).mean(-1)


class SEBlock(nn.Module):
    """(N,C,H,W) -> (N,C,H,W)"""

    def __init__(self, in_channel, r):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_channel, in_channel // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // r, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return x * se_weight  # (N, C, H, W)


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_layer, use_bias):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)


class UnetDoubleConvBlock(nn.Module):
    """
        Unet double Conv block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetDoubleConvBlock, self).__init__()

        self.mode = mode

        if self.mode == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'bottleneck':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'res-bottleneck':
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
            out_sequence = [
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        if self.mode == 'res-bottleneck':
            x_ = self.projection(x)
            out = self.out_block(x_ + self.bottleneck(x_))
        else:
            out = self.out_block(self.model(x))
        return out


class UnetDownsamplingBlock(nn.Module):
    """
        Unet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_conv, mode='default'):
        super(UnetDownsamplingBlock, self).__init__()

        downsampling_layers = list()
        if use_conv:
            downsampling_layers += [
                nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(in_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            downsampling_layers += [nn.MaxPool2d(2)]

        self.model = nn.Sequential(
            nn.Sequential(*downsampling_layers),
            UnetDoubleConvBlock(in_channel, out_channel, norm_layer, use_dropout, use_bias, mode=mode)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UnetUpsamplingBlock(nn.Module):
    """
        Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        out = torch.cat([x2, self.upsample(x1)], dim=1)
        out = self.double_conv(out)
        return out


class UnetBackbone(nn.Module):
    """
        Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=True, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(UnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                UnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


class DualDomainPriorFusionBlock(nn.Module):
    """
        prior_i:(1, H, W) prior_p:(1, H, W) -->  (channel, H, W)
    """

    def __init__(self, channel, norm_layer, use_bias):
        super(DualDomainPriorFusionBlock, self).__init__()
        self.prior_i_feature_extraction = nn.Sequential(
            nn.Conv2d(1, channel // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.prior_p_feature_extraction = nn.Sequential(
            nn.Conv2d(1, channel // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(channel // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.attn_i2p = AttentionBlock(channel // 2, channel // 2, channel // 2, norm_layer, use_bias)
        self.attn_p2i = AttentionBlock(channel // 2, channel // 2, channel // 2, norm_layer, use_bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, bias=use_bias)
        )

    def forward(self, prior_i, prior_p):
        prior_i_feature = self.prior_i_feature_extraction(prior_i)
        prior_p_feature = self.prior_p_feature_extraction(prior_p)
        attn_i2p_feature = self.attn_i2p(prior_i_feature, prior_p_feature)
        attn_p2i_feature = self.attn_p2i(prior_p_feature, prior_i_feature)
        mutual_attn_feature = torch.cat([attn_i2p_feature, attn_p2i_feature], dim=1)
        prior = self.fusion(mutual_attn_feature)
        return prior


class TextureGuidedDemodulationBlock(nn.Module):
    """
        x:(in_channel, H, W) texture:(texture_channel, H, W) -->  (in_channel, H, W)
    """

    def __init__(self, in_channel, texture_channel, norm_layer, use_bias):
        super(TextureGuidedDemodulationBlock, self).__init__()
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
        )
        self.texture_block = nn.Sequential(
            nn.Conv2d(texture_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.multiplier_block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
            nn.Sigmoid()
        )
        self.bias_block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, texture):
        x_feature = self.in_block(x)
        texture_feature = self.texture_block(texture)
        fused_feature = self.fusion(torch.cat([x_feature, texture_feature], dim=1))
        multiplier = self.multiplier_block(fused_feature)
        bias = self.bias_block(fused_feature)
        out = x * multiplier + bias
        return out
