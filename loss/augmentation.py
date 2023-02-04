import random

import torch
import torch.nn as nn
from einops import rearrange

def rand_spatial_seed():
    return {
        "flipX" : random.random() < 0.37,
        "flipY" : random.random() < 0.37,
        "reverseXY" : random.random() < 0.37
    }

def rand_spatial_apply(tensor, seed = None):
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
