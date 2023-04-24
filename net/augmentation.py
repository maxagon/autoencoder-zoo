import random

import torch
from einops import rearrange

import torch.nn.functional as F
import torchvision.transforms as transforms


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = (
        torch.linspace(-1.0, 1.0, W, device=img.device)
        .view(1, 1, 1, W)
        .expand(B, -1, H, -1)
    )
    yy = (
        torch.linspace(-1.0, 1.0, H, device=img.device)
        .view(1, 1, H, 1)
        .expand(B, -1, -1, W)
    )
    grid = torch.cat([xx, yy], 1)
    flow_ = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        1,
    )
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


def rand_distortion_map(img, std_in_pixels, upscale_factor=0):
    noise_shape = (
        1,
        2,
        img.shape[2] // upscale_factor,
        img.shape[3] // upscale_factor,
    )
    if upscale_factor == 0:
        upscale_ratio = 1
    else:
        upscale_ratio = 2.0 ** (upscale_factor - 1)

    noise_list = []
    for i in range(img.shape[0]):
        if random.random() < 0.15:
            noise = torch.zeros(size=noise_shape, device=img.device)
        else:
            noise = torch.normal(
                mean=0.0,
                std=random.random() * std_in_pixels / upscale_ratio,
                size=noise_shape,
                device=img.device,
            )
        noise_list.append(noise)
    noise = torch.cat(noise_list, dim=0)
    noise = transforms.transforms.F.gaussian_blur(noise, kernel_size=[5, 5])
    if upscale_factor == 0:
        return noise
    return (
        F.interpolate(
            noise, scale_factor=upscale_ratio, antialias=True, mode="bilinear"
        )
        * upscale_ratio
    )


def upscale_distortion_map(distortion):
    return (
        F.interpolate(distortion, scale_factor=2.0, antialias=True, mode="bilinear")
        * 2.0
    )


def rand_distortion_apply(img, distortion):
    return warp(img=img, flow=distortion)


def rand_spatial_seed(batch_size=1):
    result = []
    for i in range(batch_size):
        result.append(
            {
                "flipX": random.random() < 0.37,
                "flipY": random.random() < 0.37,
                "reverseXY": random.random() < 0.37,
            }
        )
    return result


def rand_spatial_apply(tensor: torch.Tensor, seed=None):
    b, _, _, _ = tensor.shape
    if seed == None:
        seed = rand_spatial_seed(batch_size=b)
    result = []
    for i in range(b):
        out_t = tensor[i].unsqueeze(0)
        if seed[i]["flipX"]:
            out_t = torch.flip(out_t, [2])
        if seed[i]["flipY"]:
            out_t = torch.flip(out_t, [3])
        if seed[i]["reverseXY"]:
            out_t = rearrange(out_t, "b c x y -> b c y x")
        result.append(out_t)
    return torch.cat(result, dim=0)


def zero_rand_element(arr):
    indexes = []
    for i in range(len(arr)):
        if arr[i] != None:
            indexes.append(i)
    assert len(indexes) >= 1
    b, _, _, _ = arr[indexes[0]].shape
    for i in range(b):
        rand = random.randint(0, len(indexes))
        if rand != len(indexes):
            arr[indexes[rand]][i] = arr[indexes[rand]][i] * 0.0
    return arr


@torch.no_grad()
def channelwise_noise_like(tensor: torch.Tensor):
    b, s, _, _ = tensor.shape
    result_rand = []
    for i in range(b):
        b_latent = []
        for si in range(s):
            target_slice = tensor[i][si]
            min_lat = torch.min(target_slice)
            max_lat = torch.max(target_slice)
            lat_slice_conv = (target_slice - min_lat) / (max_lat - min_lat + 0.00001)
            mean = torch.mean(lat_slice_conv)
            std = torch.std(lat_slice_conv)
            if std == 0 or torch.isnan(std):
                new_lat = torch.full_like(target_slice, mean)
            else:
                new_lat = torch.normal(
                    mean, std, target_slice.shape, device=tensor.device
                )
                new_lat = new_lat * (max_lat - min_lat) + min_lat
            b_latent.append(new_lat.unsqueeze(0).unsqueeze(0))
        merged = torch.cat(b_latent, dim=1)
        result_rand.append(merged)
    return torch.cat(result_rand, dim=0)
