import torch.utils.data as data
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import os
import pathlib
import random

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    print("img", len(images))
    return images


def load_any_image(path, isA):
    suffix = pathlib.Path(path).suffix
    if is_image_file(path):
        return Image.open(path).convert("RGB")
    if suffix != ".npy":
        print("Unknown image format:", suffix)
        return None
    return torch.FloatTensor(np.load(path)[0])


class SingleDataset(data.Dataset):
    def __init__(self, path, img_size=256, resize_size=400):
        super(SingleDataset, self).__init__()

        self.dir_A = os.path.join(path, "trainA")
        self.resize_size = resize_size
        self.A_paths = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.img_size = img_size
        self.random_crop = transforms.Compose(
            [transforms.RandomCrop(size=(self.img_size, self.img_size))]
        )

    def apply_trans(self, img):
        img_t = self.transform(img)
        shape = img_t.size()  # chanels, sizeX, sizeY
        resize_val = min(shape[1], shape[2])
        resize_val_r = max(int(resize_val * random.random()), self.resize_size)
        resize_ratio = resize_val / resize_val_r
        img_t = transforms.transforms.F.resize(
            img_t,
            [
                max(int(shape[1] / resize_ratio), self.img_size),
                max(int(shape[2] / resize_ratio), self.img_size),
            ],
            antialias=True,
        )
        return self.random_crop(img_t)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = self.apply_trans(load_any_image(A_path, True))
        return {"A0": A, "A_paths": A_path}

    def __len__(self):
        return self.A_size * 1000
