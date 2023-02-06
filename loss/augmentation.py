import random

import torch
from einops import rearrange

def rand_spatial_seed():
    return {
        "flipX" : random.random() < 0.37,
        "flipY" : random.random() < 0.37,
        "reverseXY" : random.random() < 0.37
    }

def rand_spatial_apply(tensor : torch.Tensor, seed = None):
    if seed == None:
        seed = rand_spatial_seed()
    out_t = tensor
    if seed["flipX"]:
        out_t = torch.flip(out_t, [2])
    if seed["flipY"]:
        out_t = torch.flip(out_t, [3])
    if seed["reverseXY"]:
        out_t = rearrange(out_t, "b c x y -> b c y x")
    return out_t

@torch.no_grad()
def channelwise_noise_like(tensor : torch.Tensor):
    b, s, _, _ = tensor
    result_rand = []
    for i in range(b):
        b_latent = []
        for si in range(s):
            target_slice = tensor[i][si]
            min_lat = torch.min(target_slice)
            max_lat = torch.max(target_slice)
            lat_slice_conv = (target_slice - min_lat) / (max_lat - min_lat)
            mean = torch.mean(lat_slice_conv)
            std = torch.std(lat_slice_conv)
            new_lat = torch.normal(mean, std, target_slice.shape).to(tensor.device)
            new_lat = new_lat * (max_lat - min_lat) + min_lat
            b_latent.append(new_lat)
        merged = torch.cat(b_latent, dim=1)
        result_rand.append(merged)
    return torch.cat(result_rand, dim=0)
