from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import visdom

VisdomExceptionBase = ConnectionError


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = torch.clamp(image_tensor[0], -1.0, 1.0).cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = ((np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0) * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


class Visualizer:
    def __init__(self, name, display_winsize=256):
        self.name = name
        self.resize = transforms.Compose(
            [transforms.Resize((display_winsize, display_winsize))]
        )
        self.img_size = display_winsize
        self.tab_id_counter = 0
        self.tab_id_reserve = 2
        self.tab_id_dict = {}
        self.vis = visdom.Visdom(server="http://localhost", port="8097", env="main")

    def get_id_for_tab_name(self, tab_name):
        if not tab_name in self.tab_id_dict:
            self.tab_id_dict[tab_name] = self.tab_id_counter
            self.tab_id_counter += self.tab_id_reserve
        return self.tab_id_dict[tab_name]

    def display_dict(self, visuals, ncols, name):
        ncols = min(ncols, len(visuals))
        title = name
        label_html = ""
        label_html_row = ""
        images = []
        idx = 0
        for label, image in visuals.items():
            decoded_img_np = None
            decoded_img_np = tensor2im(self.resize(image))
            label_html_row += "<td>%s</td>" % label
            images.append(decoded_img_np.transpose([2, 0, 1]))
            idx += 1
            if idx % ncols == 0:
                label_html += "<tr>%s</tr>" % label_html_row
                label_html_row = ""
        if label_html_row != "":
            label_html += "<tr>%s</tr>" % label_html_row
        try:
            tab_id = self.get_id_for_tab_name(name)
            self.vis.images(
                images,
                nrow=ncols,
                win=tab_id,
                padding=2,
                opts=dict(title=title + " images"),
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def display_losses(self, iter, losses, name):
        off_str = "plot_data_" + name
        if not hasattr(self, off_str):
            setattr(self, off_str, {"X": [], "Y": [], "legend": list(losses.keys())})
        plot_data = getattr(self, off_str)
        plot_data["X"].append(iter)
        plot_data["Y"].append([float(losses[k]) for k in losses.keys()])
        try:
            tab_id = self.get_id_for_tab_name(name)
            self.vis.line(
                X=np.stack([np.array(plot_data["X"])] * len(plot_data["legend"]), 1),
                Y=np.array(plot_data["Y"]),
                opts={
                    "title": name,
                    "legend": plot_data["legend"],
                    "xlabel": "iter",
                    "ylabel": "loss",
                },
                win=tab_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()
