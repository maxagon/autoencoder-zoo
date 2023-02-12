import math

import torch
import torch.nn as nn

import blocks.feedforward as ff
import blocks.spatial as spatial
import blocks.nonlinear as nl
import ae_base as ae
import blocks.unet as UNet

def ImgDimReduceMini_360k(pretrained=True):
    model = ImgDimReduceMini(
        in_out_dim=3,
        ae_lat_dim=8,
        ae_depth=[1, 1, 2],
        ae_dim=[16, 32, 48],
        unet_depth=[1, 1, 1],
        unet_dim=[16, 32, 48])
    if pretrained:
        model.load_state_dict(torch.load("pretrained/ImgDimReduceMini_360k_laion_aestheticsv2_6.5.bin"))
    return model

class SimpleResBlock(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        modules = []
        for i in range(n_layers):
            modules.append(ff.Conv2DBlock(in_dim=dim, out_dim=dim, kernel_rad=1, pad_type='zero', bias=True, nonlinearity=nl.ReLU()))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        out = x
        out = self.model(out)
        out = (x + out) * (1.0 / math.sqrt(2.0))
        return out

class FactoryAE(UNet.UNetFactory):
    def __init__(self):
        super().__init__()

    def make_downscale(self, in_dim, out_dim):
        return spatial.ConvolutionDownscale(in_dim, out_dim)

    def make_upscale(self, in_dim, out_dim):
        return spatial.ConvolutionUpscale(in_dim, out_dim)

    def make_feedforward(self, dim, n_layers):
        return SimpleResBlock(dim=dim, n_layers=n_layers)

    def make_dim_convert(self, in_dim, out_dim):
        return ff.Conv2DBlock(in_dim=in_dim, out_dim=out_dim, kernel_rad=0, bias=True)

class FactoryUNet(UNet.UNetFactory):
    def __init__(self):
        super().__init__()

    def make_downscale(self, in_dim, out_dim):
        return spatial.ConvolutionDownscale(in_dim, out_dim)

    def make_upscale(self, in_dim, out_dim):
        return spatial.LanczosUpscale(in_dim, out_dim)

    def make_feedforward(self, dim, n_layers):
        modules = []
        for i in range(n_layers):
            modules.append(ff.Conv2DBlock(in_dim=dim, out_dim=dim, kernel_rad=1, pad_type='zero', bias=True, nonlinearity=nl.ReLU()))
        return nn.Sequential(*modules)

    def make_dim_convert(self, in_dim, out_dim):
        return ff.Conv2DBlock(in_dim=in_dim, out_dim=out_dim, kernel_rad=0, bias=False)

class ImgDimReduceMini(ae.AEBase):
    def __init__(self, in_out_dim, ae_depth, ae_dim, ae_lat_dim, unet_depth, unet_dim, img_size):
        super().__init__()

        r_ae_depth = list(reversed(ae_depth))
        r_ae_dim = list(reversed(ae_dim))

        unet_factory = FactoryUNet()
        ae_factory = FactoryAE()

        self.rec = UNet.UnetReconstruction(unet_factory, unet_depth, unet_dim)
        self.down_blocks = ae_factory.unet_downscale_blocks(dims=ae_dim, depth=ae_depth)
        self.up_blocks = ae_factory.unet_upscale_blocks(dims=r_ae_dim, depth=r_ae_depth)
        self.from_img = ae_factory.make_dim_convert(in_dim=in_out_dim, out_dim=ae_dim[0])
        self.to_lat = ae_factory.make_dim_convert(in_dim=ae_dim[-1], out_dim=ae_lat_dim)
        self.from_lat = ae_factory.make_dim_convert(in_dim=ae_lat_dim, out_dim=r_ae_dim[0])
        self.to_img = ae_factory.make_dim_convert(in_dim=r_ae_dim[-1], out_dim=in_out_dim)

    def encode_features(self, input):
        out = input
        out = self.from_img(out)
        for b in self.down_blocks:
            out = b(out)
        return out

    def encode_bottleneck(self, input):
        out = input
        out = self.to_lat(out)
        out = torch.clamp(out, -1.0, 1.0)
        return out

    def decode_expand(self, input):
        out = input
        out = self.from_lat(out)
        return out

    def decode_features(self, input):
        out = input
        for b in self.up_blocks:
            out = b(out)
        out = self.rec(out)
        out = self.to_img(out)
        return out
