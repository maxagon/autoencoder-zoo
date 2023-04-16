import math

import torch
import torch.nn as nn

import net
import model


class SimpleResBlock(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        modules = []
        for i in range(n_layers):
            modules.append(
                net.Conv2DBlock(
                    in_dim=dim,
                    out_dim=dim,
                    kernel_rad=1,
                    pad_type="zero",
                    bias=True,
                    nonlinearity=net.ReLU(),
                )
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = x
        out = self.model(out)
        out = (x + out) * (1.0 / math.sqrt(2.0))
        return out


class FactoryAE(net.UNetFactory):
    def __init__(self):
        super().__init__()

    def make_downscale(self, in_dim, out_dim):
        return net.ConvolutionDownscale(in_dim, out_dim)

    def make_upscale(self, in_dim, out_dim):
        return net.ConvolutionUpscale(in_dim, out_dim)

    def make_feedforward(self, dim, n_layers, heads):
        assert heads == 0
        return SimpleResBlock(dim=dim, n_layers=n_layers)

    def make_dim_convert(self, in_dim, out_dim):
        return net.Conv2DBlock(in_dim=in_dim, out_dim=out_dim, kernel_rad=0, bias=True)


class FactoryUNet(net.UNetFactory):
    def __init__(self):
        super().__init__()

    def make_downscale(self, in_dim, out_dim):
        return net.ConvolutionDownscale(in_dim, out_dim)

    def make_upscale(self, in_dim, out_dim):
        return net.LanczosUpscale(in_dim, out_dim)

    def make_feedforward(self, dim, n_layers, heads):
        assert heads == 0
        modules = []
        for i in range(n_layers):
            modules.append(
                net.Conv2DBlock(
                    in_dim=dim,
                    out_dim=dim,
                    kernel_rad=1,
                    pad_type="zero",
                    bias=True,
                    nonlinearity=net.ReLU(),
                )
            )
        return nn.Sequential(*modules)

    def make_dim_convert(self, in_dim, out_dim):
        return net.Conv2DBlock(in_dim=in_dim, out_dim=out_dim, kernel_rad=0, bias=False)


class PyramidDimConvert(nn.Module):
    def __init__(self, in_dims, out_dims, factory: net.UNetFactory):
        super().__init__()
        assert len(in_dims) == len(out_dims)
        blocks = []
        for i in range(len(in_dims)):
            blocks.append(
                factory.make_dim_convert(in_dim=in_dims[i], out_dim=out_dims[i])
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = []
        for i in range(len(x)):
            out.append(self.blocks[i](x[i]))
        return out


class PyramidEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        dim_shared,
        depth_shared,
        dim_pyramid,
        depth_pyramid,
        factory: net.UNetFactory,
    ):
        super().__init__()
        self.from_img = net.Conv2DBlock(
            in_dim=in_dim, out_dim=dim_shared[0], kernel_rad=1, pad_type="zero"
        )
        self.from_img_pyramid = net.Conv2DBlock(
            in_dim=in_dim, out_dim=dim_pyramid[0], kernel_rad=1, pad_type="zero"
        )

        self.shared_cascade = factory.unet_downscale_blocks(
            dims=dim_shared, depth=depth_shared, heads=[0] * len(depth_shared)
        )
        self.cascade_len = len(dim_pyramid)
        self.shared_cascade_len = len(dim_shared)

        cur_dim = 0
        shared_out_dims = []
        for i in range(len(dim_shared)):
            if i <= 2:
                cur_dim = cur_dim + dim_shared[i]
            shared_out_dims.append(cur_dim)

        cat_convert = []
        for i in range(0, len(dim_pyramid)):
            blocks = []
            dim_in = dim_pyramid[i] + shared_out_dims[i]
            dim_out = dim_pyramid[i]
            blocks.append(factory.make_dim_convert(in_dim=dim_in, out_dim=dim_out))
            cat_convert.append(nn.Sequential(*blocks))
        self.cat_convert = nn.ModuleList(cat_convert)

        pyramid_cascade = []
        for i in range(self.cascade_len):
            blocks = []
            dim_in = dim_pyramid[i]
            blocks.append(
                factory.make_feedforward(dim=dim_in, n_layers=depth_pyramid[i], heads=0)
            )
            if i != self.cascade_len - 1:
                blocks.append(factory.make_downscale(dim_in, dim_pyramid[i + 1]))
            pyramid_cascade.append(nn.Sequential(*blocks))
        self.pyramid_cascade = nn.ModuleList(pyramid_cascade)

    def forward(self, x):
        features = [None] * self.cascade_len
        imgs_down = []
        cur_img = x
        for i in range(self.cascade_len):
            imgs_down.append(cur_img)
            cur_img = nn.functional.avg_pool2d(input=cur_img, kernel_size=2)

        for i in range(self.cascade_len):
            cur_img = self.from_img(imgs_down[i])
            for i2 in range(self.shared_cascade_len):
                index = i + i2
                if index < self.cascade_len:
                    cur_img = self.shared_cascade[i2](cur_img)
                    if features[index] == None:
                        features[index] = cur_img
                    else:
                        features[index] = torch.cat([features[index], cur_img], dim=1)

        out = self.from_img_pyramid(x)
        for i in range(self.cascade_len):
            out = torch.cat([features[i], out], dim=1)
            out = self.cat_convert[i](out)
            out = self.pyramid_cascade[i](out)
        return out


class ImgDimReduceMini(model.AEBase):
    def __init__(self, in_out_dim, ae_depth, ae_dim, ae_lat_dim, unet_depth, unet_dim):
        super().__init__()

        r_ae_depth = list(reversed(ae_depth))
        r_ae_dim = list(reversed(ae_dim))

        unet_factory = FactoryUNet()
        ae_factory = FactoryAE()

        self.rec = net.UnetReconstruction(unet_factory, unet_depth, unet_dim)
        self.down_block = PyramidEncoder(
            in_dim=3,
            dim_shared=ae_dim,
            depth_shared=ae_depth,
            dim_pyramid=ae_dim,
            depth_pyramid=ae_depth,
            factory=ae_factory,
        )

        self.up_blocks = ae_factory.unet_upscale_blocks(
            dims=r_ae_dim, depth=r_ae_depth, heads=[0] * len(ae_dim)
        )
        self.to_lat = ae_factory.make_dim_convert(in_dim=ae_dim[-1], out_dim=ae_lat_dim)
        self.from_lat = ae_factory.make_dim_convert(
            in_dim=ae_lat_dim, out_dim=r_ae_dim[0]
        )
        self.to_img = ae_factory.make_dim_convert(
            in_dim=r_ae_dim[-1], out_dim=in_out_dim
        )

    @classmethod
    def model_rgb_4_4_8(cls):
        model_rgb_4_4_8 = ImgDimReduceMini(
            in_out_dim=3,
            ae_lat_dim=8,
            ae_depth=[1, 2, 4],
            ae_dim=[16, 32, 48],
            unet_depth=[1, 2, 4],
            unet_dim=[16, 32, 48],
        )
        return model_rgb_4_4_8

    def encode_features(self, input):
        out = input
        out = self.down_block(out)
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
