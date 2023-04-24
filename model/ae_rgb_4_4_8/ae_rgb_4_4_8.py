from .model import ImgDimReduceMini
import training
import torch


def model_rgb_4_4_8(pretrained=False):
    model_rgb_4_4_8 = ImgDimReduceMini(
        in_out_dim=3,
        ae_lat_dim=8,
        ae_depth=[1, 1, 2],
        ae_dim=[16, 32, 48],
        unet_depth=[1, 2, 4],
        unet_dim=[16, 32, 48],
    )
    if pretrained:
        model_rgb_4_4_8.load_state_dict(
            torch.load(training.get_pretrained_path("model_rgb_4_4_8.bin"))
        )
    return model_rgb_4_4_8


def model_rgb_4_4_8_trainer(device, checkpoint_dir, dataset_dir):
    model = model_rgb_4_4_8()
    schedule = [
        (
            0,
            training.AeHyperScheduleElement(
                resolution=280,
                batch_size=8,
                optimizer="adam",
                latent_gauss_dropout=0.05,
                use_aestetics_loss=False,
                use_lpips_loss=True,
                use_dinov2_loss=False,
                use_l2_loss=True,
                use_warp_augmentation=False,
                use_spatial_augmentation=True,
            ),
        ),
    ]
    lr_params = training.LrScheduleParams(
        warmup_steps=20000,
        cosine_ampl=0.2,
        cosine_freq=1000,
        exp_decay=45000,
        lr_min=0.0000001,
        lr_max=0.0005,
    )
    return training.AeTrainer(
        checkpoint_folder_name=checkpoint_dir,
        device=device,
        dataset_dir=dataset_dir,
        schedule=schedule,
        lr_schedule_params=lr_params,
    )
