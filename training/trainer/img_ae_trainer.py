import torch
import torch.nn as nn
import torchvision.datasets as datasets
import net
import training
import math
from dataclasses import dataclass
import torchvision.transforms as transforms
import model
import numpy as np
from .img_ae_validate import Validate


@dataclass(frozen=True)
class AeHyperScheduleElement:
    resolution: int
    batch_size: int
    use_aestetics_loss: bool
    use_lpips_loss: bool
    use_dinov2_loss: bool
    use_l2_loss: bool
    use_warp_augmentation: bool
    use_spatial_augmentation: bool
    latent_gauss_dropout: float
    optimizer: str


@dataclass(frozen=True)
class LrScheduleParams:
    warmup_steps: int
    cosine_ampl: float
    cosine_freq: float
    exp_decay: float
    lr_min: float
    lr_max: float


class TrainingSession:
    def __init__(
        self,
        device,
        model: model.ae_base,
        schedule_element: AeHyperScheduleElement,
        iteration: int,
        optimizer: str,
        lr_schedule_params: LrScheduleParams,
    ) -> None:
        print("Training model:", type(model))
        self.device = device
        self.schedule_element = schedule_element
        self.model = model.to(device)
        self.model_params = list(self.model.parameters())
        self.lr_schedule_params = lr_schedule_params
        total_params = sum(p.numel() for p in self.model_params if p.requires_grad)
        print("Total params:", total_params)

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                [p for p in self.model_params if p.requires_grad],
                lr=self.get_lr(iter=iteration),
                betas=(0.5, 0.999),
                weight_decay=0.0001,
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [p for p in self.model_params if p.requires_grad],
                lr=self.get_lr(iter=iteration),
                weight_decay=0.0001,
            )

        if self.schedule_element.use_l2_loss:
            self.l2_loss = nn.MSELoss().to(device)

        self.lpips_loss = net.LPIPS().eval().to(device)

        if self.schedule_element.use_dinov2_loss:
            self.dino_loss = net.VitDinoV2().to(device)

        if self.schedule_element.use_aestetics_loss:
            self.aestetics_loss = net.AesteticScoreLoss().eval().to(device)

        self.upscale = net.Upscale()
        self.distortion_std = 3

        if self.schedule_element.latent_gauss_dropout != 0.0:
            self.gauss_vae = net.GaussianDropout(
                dropout=self.schedule_element.latent_gauss_dropout
            )

    def get_lr(self, iter):
        if iter <= self.lr_schedule_params.warmup_steps:
            return self.lr_schedule_params.lr_max
        else:
            step = iter - self.lr_schedule_params.warmup_steps
            lr = self.lr_schedule_params.lr_min + (
                self.lr_schedule_params.lr_max - self.lr_schedule_params.lr_min
            ) * (
                1.0
                + self.lr_schedule_params.cosine_ampl
                * math.cos(math.pi * 2.0 * step / self.lr_schedule_params.cosine_freq)
            ) * math.exp(
                -step / self.lr_schedule_params.exp_decay
            )
            return lr

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train_iteration(self, dataset_iterator, iter):
        img = next(dataset_iterator).to(self.device)
        b = img.shape[0]
        latent = self.model.encode(img)

        if self.schedule_element.use_spatial_augmentation:
            seed = net.rand_spatial_seed(b)
            img = net.rand_spatial_apply(img, seed)
            latent = net.rand_spatial_apply(latent, seed)

        if self.schedule_element.use_warp_augmentation:
            distortion = net.rand_distortion_map(
                img=latent, std_in_pixels=self.distortion_std, upscale_factor=2
            )
            latent = net.rand_distortion_apply(img=latent, distortion=distortion)

            distortion_img_space = net.upscale_distortion_map(
                net.upscale_distortion_map(distortion)
            )
            img = net.rand_distortion_apply(img=img, distortion=distortion_img_space)

        if self.schedule_element.latent_gauss_dropout != 0.0:
            latent = self.gauss_vae(latent)

        img_recon = self.model.decode(latent)

        l2_loss = 0.0
        if self.schedule_element.use_l2_loss:
            l2_loss = self.l2_loss(img, img_recon)

        lpips_loss = 0.0
        if self.schedule_element.use_lpips_loss:
            lpips_loss = torch.sum(self.lpips_loss(img, img_recon)) / b

        dino_loss = 0.0
        if self.schedule_element.use_dinov2_loss:
            dino_loss = self.dino_loss(input=img_recon, target=img) * 0.15

        aestetics_loss = 0.0
        if self.schedule_element.use_aestetics_loss:
            aestetics_loss = (
                self.aestetics_loss(target_img=img, recon_img=img_recon) * 0.05
            )

        total_loss = l2_loss + lpips_loss + aestetics_loss + dino_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.get_lr(iter)
        self.set_lr(lr)

        loss_dict = {
            "lr": lr,
            "l2_loss": l2_loss,
            "lpips_loss": lpips_loss,
            "dino_loss": dino_loss,
            "aestetics_loss": aestetics_loss,
            "total_loss": total_loss,
        }

        return loss_dict

    @torch.no_grad()
    def visualize(self, img):
        latent = self.model.encode(img)
        img_recon = self.model.decode(latent)
        distortion = net.rand_distortion_map(
            img=latent, std_in_pixels=self.distortion_std, upscale_factor=2
        )

        latent_distort = net.rand_distortion_apply(img=latent, distortion=distortion)
        img_recon_distort = self.model.decode(latent_distort)

        distortion_img_space = net.upscale_distortion_map(
            net.upscale_distortion_map(distortion)
        )
        img_distort = net.rand_distortion_apply(
            img=img, distortion=distortion_img_space
        )

        noise_sample = net.channelwise_noise_like(latent)
        noise_img = self.model.decode(noise_sample)

        img_dict = {
            "img_orig": img,
            "img_recon": img_recon,
            "img_distort": img_distort,
            "img_recon_distort": img_recon_distort,
            "img_sampled_from_noise": noise_img,
        }

        latent = self.upscale(self.upscale(latent))
        for i in range(8):
            img_dict["latent_space" + str(i)] = latent[:, i : i + 1, :, :]
        latent_distort = self.upscale(self.upscale(latent_distort))
        for i in range(8):
            img_dict["latent_space_distort" + str(i)] = latent_distort[
                :, i : i + 1, :, :
            ]

        return img_dict

    @torch.no_grad()
    def calculate_metrics(self, img):
        img_recon = self.model(img)
        lpips = self.lpips_loss(img, img_recon)
        psnr = self.psnr(input=img_recon, grountruth=img)
        ssim = self.ssim(input=img_recon, grountruth=img)
        result_metrics = {"LPIPS": lpips, "PSNR": psnr, "SSIM": ssim}
        return result_metrics

    def get_val(self, val_dataset_dir, batch_size):
        return Validate(
            self.model,
            resolution=280,
            val_dataset_dir=val_dataset_dir,
            batch_size=batch_size,
            lpips=self.lpips_loss,
            device=self.device,
        )


def reformat_schedule(hyper_schedule):
    iter_arr = []
    element_arr = []
    for iter, element in hyper_schedule:
        iter_arr.append(iter)
        element_arr.append(element)
    return iter_arr, element_arr


def need_update(iter, iter_arr):
    return iter in iter_arr


def get_schedule_element_from_iter(
    iter, iter_arr, element_arr
) -> AeHyperScheduleElement:
    for i in range(len(iter_arr)):
        if iter == iter_arr[i]:
            return element_arr[i]
        if iter < iter_arr[i]:
            return element_arr[i - 1]
    return element_arr[-1]


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


class AeTrainer:
    def __init__(
        self,
        checkpoint_folder_name: str,
        device,
        dataset_dir: str,
        schedule: list[tuple[int, AeHyperScheduleElement]],
        lr_schedule_params: LrScheduleParams,
    ) -> None:
        self.iter_arr, self.element_arr = reformat_schedule(schedule)

        self.end_iter = 300000
        self.validate_each_iter = 10000
        self.safe_each = 5000
        self.device = device
        self.checkpoint_folder_name = checkpoint_folder_name
        self.dataset_dir = dataset_dir
        self.trainer_session = None
        self.lr_schedule_params = lr_schedule_params
        self.reload()

    def reload(self):
        self.iter = training.get_last_checkpoint_index(
            self.checkpoint_folder_name, ["model_state"]
        )
        self.schedule_element = get_schedule_element_from_iter(
            self.iter, self.iter_arr, self.element_arr
        )
        print(
            "Reloading model! Iteration:", self.iter, "Schedule:", self.schedule_element
        )

        if self.trainer_session != None:
            del self.trainer_session
            self.trainer_session = None

        self.trainer_session = TrainingSession(
            device=self.device,
            schedule_element=self.schedule_element,
            iteration=self.iter,
            optimizer=self.schedule_element.optimizer,
        )

        if self.iter != 0:
            model_checkpoint_path = training.get_checkpoint_file(
                self.checkpoint_folder_name, data_name="model_state", index=self.iter
            )
            self.trainer_session.model.load_state_dict(
                torch.load(model_checkpoint_path)
            )
            print("Loading model state:", model_checkpoint_path)

        optimizer_last_iter = training.get_last_checkpoint_index(
            self.checkpoint_folder_name, data_names=[self.schedule_element.optimizer]
        )
        if optimizer_last_iter == self.iter and self.iter != 0:
            optimizer_last_checkpoint = training.get_checkpoint_file(
                self.checkpoint_folder_name,
                data_name=self.schedule_element.optimizer,
                index=self.iter,
            )
            print("Loading optimizer state:", optimizer_last_checkpoint)
            self.trainer_session.optimizer.load_state_dict(
                torch.load(optimizer_last_checkpoint)
            )
        else:
            print(
                "Switching to new optimizer. Optimizer:",
                self.schedule_element.optimizer,
            )

        self.dataset = training.SingleDataset(
            self.dataset_dir,
            img_size=self.schedule_element.resolution,
            resize_size=self.schedule_element.resolution * 2,
        )
        self.dataset_loader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                self.schedule_element.batch_size,
                shuffle=True,
                num_workers=int(4),
                pin_memory=True,
            )
        )

    def checkpoint_save(self):
        model_checkpoint_name = training.get_checkpoint_file(
            self.checkpoint_folder_name, data_name="model_state", index=self.iter
        )
        print("Saving model checkpoint:", model_checkpoint_name)
        torch.save(self.trainer_session.model.state_dict(), model_checkpoint_name)
        optimizer_checkpoint_name = training.get_checkpoint_file(
            self.checkpoint_folder_name,
            data_name=self.schedule_element.optimizer,
            index=self.iter,
        )
        print("Saving optimizer checkpoint:", optimizer_checkpoint_name)
        torch.save(
            self.trainer_session.optimizer.state_dict(), optimizer_checkpoint_name
        )

    def train_iter(self):
        self.iter = self.iter + 1
        if self.iter % self.safe_each == 0:
            self.checkpoint_save()
        if need_update(iter=self.iter, iter_arr=self.iter_arr):
            self.reload()
        loss_dict = self.trainer_session.train_iteration(self.dataset_loader, self.iter)
        return self.iter, loss_dict

    def visualize(self, img):
        return self.trainer_session.visualize(img)

    def get_val(self, val_dataset_dir):
        return self.trainer_session.get_val(
            val_dataset_dir=val_dataset_dir, batch_size=self.schedule_element.batch_size
        )
