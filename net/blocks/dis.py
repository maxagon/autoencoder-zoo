import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad

def hinge_g_loss(logits_fake):
    logits_fake = -torch.mean(logits_fake)
    return logits_fake

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def vanilla_g_loss(logits_fake):
    return -torch.mean(logits_fake)

def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images,
                           grad_outputs=list(map(lambda t: torch.ones(t.size(), device=images.device), outputs)),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc=8, ndf=128, n_layers=4):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = True

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.ReLU(),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, 3, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.loss = nn.MSELoss()
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

    def dis_loss(self, real_features, fake_features, apply_grad_penality=True):
        grad_penality = 0
        real_features.requires_grad_()
        real_logits = self.forward(real_features)
        if apply_grad_penality:
            grad_penality = gradient_penalty(real_features, (real_logits,))
        real_loss = self.loss(torch.ones_like(real_logits), real_logits)
        fake_logits = self.forward(fake_features)
        fake_loss = self.loss(torch.zeros_like(fake_logits), self.forward(fake_features))
        return 0.5 * (real_loss + fake_loss) + grad_penality

    def gen_loss(self, fake_features):
        fake_logits = self.forward(fake_features)
        fake_loss = self.loss(torch.ones_like(fake_logits), self.forward(fake_features))
        return fake_loss