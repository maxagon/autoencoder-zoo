import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import training


# https://github.com/LAION-AI/aesthetic-predictor
class AesteticScoreLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.load_state_dict(
            torch.load(training.get_checkpoint_path("ava+logos-l14-linearMSE.pth"))
        )
        self.clip_model, _ = torch.hub.load("openai/CLIP", "ViT_L_14")
        mean_expected = np.array([0.48145466, 0.4578275, 0.40821073])
        std_expected = np.array([0.26862954, 0.26130258, 0.27577711])
        mean = 2.0 * mean_expected - 1.0
        std = 2.0 * std_expected
        self.scaling = transforms.Normalize(tuple(mean.tolist()), tuple(std.tolist()))

    def calculate_score(self, x):
        out = x
        out = self.scaling(out)
        out = nn.functional.interpolate(out, size=(224, 224))
        out = self.clip_model.encode_image(out)
        out = out.to(x.dtype)
        out = self.layers(out)
        return out

    def forward(self, target_img, recon_img):
        score_recon = self.calculate_score(recon_img)
        score_target = self.calculate_score(target_img).detach()
        score_diff = score_target - score_recon
        out = nn.functional.relu(1.0 - score_diff) * 0.5
        out = torch.sum(out, dim=0) / score_diff.shape[0]
        return out
