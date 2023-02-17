import random

import torch
from einops import rearrange

def rand_spatial_seed(batch_size = 1):
    result = []
    for i in range(batch_size):
        result.append({
            "flipX" : random.random() < 0.37,
            "flipY" : random.random() < 0.37,
            "reverseXY" : random.random() < 0.37
        })
    return result

def rand_spatial_apply(tensor : torch.Tensor, seed = None):
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

@torch.no_grad()
def channelwise_noise_like(tensor : torch.Tensor):
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
                new_lat = torch.normal(mean, std, target_slice.shape).to(tensor.device)
                new_lat = new_lat * (max_lat - min_lat) + min_lat
            b_latent.append(new_lat.unsqueeze(0).unsqueeze(0))
        merged = torch.cat(b_latent, dim=1)
        result_rand.append(merged)
    return torch.cat(result_rand, dim=0)
