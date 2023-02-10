from abc import abstractmethod

import torch
import torch.nn as nn

class UNetFactory():
    def __init__(self):
        pass

    @abstractmethod
    def make_downscale(self, img_size, in_dim, out_dim):
        pass

    @abstractmethod
    def make_upscale(self, img_size, in_dim, out_dim):
        pass

    @abstractmethod
    def make_feedforward(self, img_size, dim, n_layers):
        pass

    @abstractmethod
    def make_dim_convert(self, img_size, in_dim, out_dim):
        pass

    # apply downscale dims-1 times
    # dims = 1 -> [[ResBlock]]
    # dims = 2 -> [[ResBlock], [Downscale, ResBlock]]
    def unet_downscale_blocks(self, img_size, dims, depth):
        modules = []
        assert(len(dims) == len(depth))
        cur_dim = dims[0]
        cur_img_size = img_size
        if depth[0] != 0:
            modules.append(self.make_feedforward(img_size=img_size, dim=dims[0], n_layers=depth[0]))
        for i in range(1, len(dims)):
            next_dim = dims[i]
            blocks = []
            blocks.append(self.make_downscale(img_size=cur_img_size, in_dim=cur_dim, out_dim=next_dim))
            cur_img_size = img_size // 2
            cur_dim = next_dim
            blocks.append(self.make_feedforward(img_size=cur_img_size, dim=cur_dim, n_layers=depth[i]))
            modules.append(nn.Sequential(*blocks))
        return nn.ModuleList(modules)
    # apply upscale dims-1 times
    # dims = 1 -> [[ResBlock]]
    # dims = 2 -> [[ResBlock], [Upscale, ResBlock]]
    def unet_upscale_blocks(self, img_size, dims, depth):
        modules = []
        assert(len(dims) == len(depth))
        cur_dim = dims[0]
        cur_img_size = img_size
        if depth[0] != 0:
            modules.append(self.make_feedforward(img_size=img_size, dim=dims[0], n_layers=depth[0]))
        for i in range(1, len(dims)):
            next_dim = dims[i]
            blocks = []
            blocks.append(self.make_upscale(img_size=cur_img_size, in_dim=cur_dim, out_dim=next_dim))
            cur_img_size = img_size // 2
            cur_dim = next_dim
            blocks.append(self.make_feedforward(img_size=cur_img_size, dim=cur_dim, n_layers=depth[i]))
            modules.append(nn.Sequential(*blocks))
        return nn.ModuleList(modules)
    # [ResBlock(dim*2), DimConvert(dim*2->dim), ResBlock(dim)]
    def unet_cat_block(self, img_size, dim, depth):
        modules = []
        modules.append(self.make_feedforward(img_size=img_size, dim=dim*2, n_layers=depth))
        modules.append(self.make_dim_convert(img_size=img_size, in_dim=dim*2, out_dim=dim))
        modules.append(self.make_feedforward(img_size=img_size, dim=dim, n_layers=depth))
        return nn.Sequential(*modules)

class UnetReconstruction(nn.Module):
    def __init__(self, factory : UNetFactory, depth, dim, img_size=256):
        super().__init__()

        self.depth = len(depth)
        assert(len(dim) == len(depth))

        r_depth = list(reversed(depth))
        r_dim = list(reversed(dim))

        self.down_blocks = factory.unet_downscale_blocks(img_size=img_size, dims=dim, depth=depth)
        self.up_blocks = factory.unet_upscale_blocks(img_size=16, dims=r_dim, depth=r_depth)

        cat_blocks = []
        cur_img_size = img_size
        for i in range(self.depth):
            cat_blocks.append(factory.unet_cat_block(img_size=cur_img_size, dim=dim[i], depth=depth[i]))
            cur_img_size = cur_img_size // 2
        cat_blocks.reverse()
        self.cat_blocks = nn.ModuleList(cat_blocks)

    def forward(self, x):
        out = x
        down_outs = []
        for i in range(self.depth):
            down_outs.append(out)
            out = self.down_blocks[i](out)

        out = self.up_blocks[0](out)

        for i in range(1, self.depth):
            out = self.up_blocks[i](out)
            cur_down_out = down_outs.pop()
            out = torch.cat([out, cur_down_out], dim=1)
            out = self.cat_blocks[i](out)

        return out
