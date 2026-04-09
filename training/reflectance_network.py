#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from inversion.utils.debugger import set_trace
from torch_utils.ops import bias_act
import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Reflectance Decoder
class CNNReflectanceBlock(nn.Module):
    """
    Reflectance network with input conditioned on OLAT directions
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_OLAT_dirs=False, n_OLATs=0, use_fp16=False, fp16_channels_last=False):
        super(CNNReflectanceBlock, self).__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.use_fp16 = False
        self.channels_last = (use_fp16 and fp16_channels_last)

        # Block 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Block 2
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # Block 3
        self.conv3 = conv3x3(out_channels, out_channels, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # Block 4
        self.conv4 = conv3x3(out_channels, out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        # Activation Function
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, in_feat, force_fp32=False):

        # Block 1
        out = self.conv1(in_feat)
        out = self.bn1(out)
        out = self.relu(out)
        # Block 2
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(in_feat)
        else:
            residual = in_feat
        out += residual
        out = self.relu(out)
        # Block 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # Block 4
        out = self.conv4(out)
        out = self.bn4(out)
        # out = self.relu(out)

        return out


class CNNEncoder(nn.Module):
    def __init__(self, block: nn.Module, in_channels: int, out_channels: int, use_OLAT_dirs: bool, n_OLATs: int):
        super().__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock = self.make_layer(block, self.in_channels, self.out_channels)

    def make_layer(self, block, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, self.out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.use_OLAT_dirs, self.n_OLATs))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, features):
        out = self.resblock(features)
        return out

# ----------------------------------------------------------------------------


class SRHighResEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv1 = conv3x3(in_channels, out_channels, stride)
        # self.bn1 = nn.BatchNorm2d(out_channels)

        self.net = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )

    def forward(self, o_feat, g_feat):
        in_feats = torch.cat([o_feat, g_feat], dim=1)
        out = self.net(in_feats)
        return out

# ----------------------------------------------------------------------------


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class ReflectanceDecoder(torch.nn.Module):
    def __init__(self, in_features, options, act_fn, depth=1, use_viewdirs=False):
        super().__init__()
        self.hidden_dim = 64
        self.act_fn = act_fn
        self.depth = depth
        self.use_viewdirs = use_viewdirs
        
        in_features = in_features + 3 if use_viewdirs else in_features

        if depth == 2:
            self.net = torch.nn.Sequential(
                FullyConnectedLayer(in_features, self.hidden_dim*2, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim*2, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
            )
        elif depth == 1:
            self.net = torch.nn.Sequential(
                FullyConnectedLayer(in_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
            )
        else:
            raise NotImplementedError

    def forward(self, sampled_features, ray_directions):
        # HACK: Hardcoding
        self.use_g_feat = False
        # Aggregate features
        if self.depth == 1:
            if not self.use_g_feat:
                _ , sampled_o_features = sampled_features
                sampled_features = sampled_o_features.mean(1)
                x = sampled_features
            else:
                sampled_g_features, sampled_o_features = sampled_features
                combined_sampled_features = torch.cat([sampled_g_features.mean(1), sampled_o_features.mean(1)], dim=2)
                x = combined_sampled_features
        elif self.depth == 2:
            if not self.use_g_feat:
                _ , sampled_o_features = sampled_features
                sampled_features = sampled_o_features.mean(1)
                x = sampled_features
            else:
                sampled_g_features, sampled_o_features = sampled_features
                combined_sampled_features = torch.cat([sampled_g_features.mean(1), sampled_o_features.mean(1)], dim=2)
                x = combined_sampled_features
        else:
            raise NotImplementedError

        N, M, C = x.shape
        x = x.view(N * M, C)
        # FIXME: Remove this exception handling
        try:
            self.use_viewdirs
        except AttributeError:
            self.use_viewdirs = False
            
        if self.use_viewdirs:
            view_dirs = ray_directions.view(N*M, 3)
            x = torch.cat((x, view_dirs), dim=1)

        # TODO: USE VIEW CONDITIONING
        x = self.net(x)
        x = x.view(N, M, -1)

        if self.act_fn == 'sigmoid':
            rgb = torch.sigmoid(x) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        elif self.act_fn == 'exp':
            x = torch.clamp_max(x, 88.)  # exp(89) = inf
            rgb = torch.exp(x)
        else:
            raise NotImplementedError

        return {'rgb': rgb}


# ------------------------------------------------------------------------
if __name__ == '__main__':
    print("Debugging Reflectance Encoder ...")
    # Create an instance of the network
    model = CNNEncoder(block=CNNReflectanceBlock, in_channels=96 + 3, out_channels=96, use_OLAT_dirs=True, n_OLATs=1).cuda()
    # model = UnetEncoder(block=ReflectanceUNetBlock, in_channels=96 + 3, out_channels=96, use_OLAT_dirs=True, n_OLATs=1).cuda()
    # model = ReflectanceTriplaneViTNetwork().cuda()
    # model = HourglassNetwork(depth=3).cuda()

    # Create a dummy input tensor with dimensions Nx256x64x64
    omega = torch.randn(4,3).cuda()
    OLAT_dirs_planes = omega[:, :, None, None].tile(1, 1, 256, 256) # N x dirs x H x W
    dummy_input = torch.randn(4, 96+3, 256, 256).cuda()
    dummy_F = torch.randn(4, 96, 256, 256).cuda()

    # Test the forward pass
    print(model)
    output = model(dummy_input)
    print(output.shape)
    print("Finished Reflectance Encoder!")
