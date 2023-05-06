import torch
import torch.nn as nn
import torchvision.datasets as datasets
import net
import torchvision.transforms as transforms
import model
import numpy as np
from tqdm import tqdm


class MetricAccumulator:
    def __init__(self) -> None:
        self.vals = []

    def accumulate(self, metric: torch.Tensor):
        assert len(metric.shape) == 1
        for i in range(metric.shape[0]):
            self.vals.append(float(metric[i]))

    def calculate(self):
        nparr = np.array(self.vals)
        metric_percentile_95 = float(np.percentile(nparr, 95))
        metric_mean = float(np.mean(nparr))
        metric_std = float(np.std(nparr))
        metric_min = float(np.min(nparr))
        metric_max = float(np.max(nparr))
        return {
            "percentile_95": metric_percentile_95,
            "mean": metric_mean,
            "std": metric_std,
            "min": metric_min,
            "max": metric_max,
        }


class Validate:
    def __init__(
        self,
        val_model: model.AEBase,
        resolution,
        val_dataset_dir,
        batch_size,
        device,
        lpips=None,
        resize=True,
    ) -> None:
        self.val_model = val_model
        dataset = datasets.ImageFolder(
            val_dataset_dir,
            transforms.Compose(
                [
                    transforms.Resize(resolution),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            if resize
            else transforms.Compose(
                [
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        self.samples = len(dataset)
        self.batch_size = batch_size
        self.val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.pnsr = net.PSNR()
        self.ssim = net.SSIM()
        self.lpips = lpips
        if self.lpips == None:
            self.lpips = net.LPIPS().to(device).eval()
        self.device = device

    @torch.no_grad()
    def validate(self):
        print("Start validation")
        pnsr_acc = MetricAccumulator()
        ssim_acc = MetricAccumulator()
        lpips = MetricAccumulator()
        with tqdm(total=self.samples) as bar:
            for val_elem in self.val_loader:
                img = val_elem[0].to(self.device)
                img_recon = self.val_model(img)
                pnsr_acc.accumulate(self.pnsr(input=img_recon, grountruth=img))
                ssim_acc.accumulate(self.ssim(input=img_recon, grountruth=img))
                lpips.accumulate(self.lpips(input=img_recon, target=img)[:, 0, 0, 0])
                bar.update(self.batch_size)
        return {
            "PSNR": pnsr_acc.calculate(),
            "SSIM": ssim_acc.calculate(),
            "LPIPS": lpips.calculate(),
        }
