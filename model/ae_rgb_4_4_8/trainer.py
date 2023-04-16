from .model import ImgDimReduceMini
import torch
import torch.nn as nn
import net
import training
import math
from dataclasses import dataclass


class TrainingSession:
    def __init__(
        self, device, use_aestetic_loss: bool, iteration: int, optimizer: str
    ) -> None:
        print(
            "Training model/ae_rgb_4_4_8 optim:{0} aestetic_loss:{1}".format(
                optimizer, use_aestetic_loss
            )
        )

        self.device = device
        self.model = ImgDimReduceMini.model_rgb_4_4_8().to(device)
        self.model_params = list(self.model.parameters())
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

        self.l2_loss = nn.MSELoss().to(device)
        self.lpips_loss = net.LPIPS().eval().to(device)
        self.aestetics_loss = None
        self.upscale = net.Upscale()
        self.psnr = net.PSNR()
        self.ssim = net.SSIM()

        if use_aestetic_loss:
            self.aestetics_loss = net.AesteticScoreLoss().eval().to(device)

    def get_lr(self, iter):
        warmup_steps = 20000
        cosine_ampl = 0.2
        cosine_freq = 1000
        exp_decay = 15000
        lr_min = 0.0000001
        lr_max = 0.0005
        if iter <= warmup_steps:
            return lr_max
        else:
            step = iter - warmup_steps
            lr = lr_min + (lr_max - lr_min) * (
                1.0 + cosine_ampl * math.cos(math.pi * 2.0 * step / cosine_freq)
            ) * math.exp(-step / exp_decay)
            return lr

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train_iteration(self, dataset_iterator, iter):
        img = next(dataset_iterator).to(self.device)
        b = img.shape[0]
        latent = self.model.encode(img)

        seed = net.rand_spatial_seed(b)
        img = net.rand_spatial_apply(img, seed)
        latent = net.rand_spatial_apply(latent, seed)

        img_recon = self.model.decode(latent)

        l2_loss = self.l2_loss(img, img_recon)
        lpips_loss = torch.sum(self.lpips_loss(img, img_recon)) / b
        aestetics_loss = 0.0
        if self.aestetics_loss != None:
            aestetics_loss = self.aestetics_loss(target_img=img, recon_img=img_recon)
        total_loss = l2_loss + lpips_loss + aestetics_loss * 0.05

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
            "aestetics_loss": aestetics_loss,
            "total_loss": total_loss,
        }

        return loss_dict

    @torch.no_grad()
    def visualize(self, img):
        latent = self.upscale(self.upscale(self.model.encode(img)))
        img_recon = self.model.decode(img)
        img_dict = {"img_orig": img, "img_recon": img_recon}
        for i in range(8):
            img_dict["latent_space" + str(i)] = latent[:, i : i + 1, :, :]
        return img_dict

    @torch.no_grad()
    def calculate_metrics(self, img):
        img_recon = self.model(img)
        lpips = self.lpips_loss(img, img_recon)
        psnr = self.psnr(input=img_recon, grountruth=img)
        ssim = self.ssim(input=img_recon, grountruth=img)
        result_metrics = {"LPIPS": lpips, "PSNR": psnr, "SSIM": ssim}
        return result_metrics


@dataclass(frozen=True)
class HyperScheduleElement:
    resolution: int
    batch_size: int
    use_aestetics_loss: bool
    optimizer: str


def reformat_schedule(hyper_schedule):
    iter_arr = []
    element_arr = []
    for iter, element in hyper_schedule:
        iter_arr.append(iter)
        element_arr.append(element)
    return iter_arr, element_arr


def need_update(iter, iter_arr):
    return iter in iter_arr


def get_schedule_element_from_iter(iter, iter_arr, element_arr) -> HyperScheduleElement:
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


class Trainer:
    def __init__(self, checkpoint_folder_name: str, device, dataset_dir: str) -> None:
        self.iter_arr, self.element_arr = reformat_schedule(
            [
                (0, HyperScheduleElement(256, 8, False, "adam")),
                (50000, HyperScheduleElement(256, 8, False, "sgd")),
                (100000, HyperScheduleElement(256, 4, True, "sgd")),
            ]
        )

        self.end_iter = 300000
        self.print_each_iter = 100
        self.validate_each_iter = 10000
        self.safe_each = 5000
        self.device = device
        self.checkpoint_folder_name = checkpoint_folder_name
        self.dataset_dir = dataset_dir
        self.trainer_session = None
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
            use_aestetic_loss=self.schedule_element.use_aestetics_loss,
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
        if self.iter % self.print_each_iter == 0:
            print(loss_dict)
