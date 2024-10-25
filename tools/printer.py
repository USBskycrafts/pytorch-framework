import matplotlib.pyplot as plt
import os
import torch
from multiprocessing import Process
from typing import Dict
import numpy as np


class Printer:
    def __init__(self, config):
        self.config = config
        self.num_worker = config.getint("output", "num_worker")
        self.path = config.get("output", "sample_path")
        self.model_name = config.get("output", "model_name")
        if not os.path.exists(f"{self.path}/{self.model_name}"):
            os.makedirs(f"{self.path}/{self.model_name}")

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)

    def launch(self, data, step):
        data = {
            modal: image.detach().cpu() for modal, image in data.items()
        }
        job = Process(
            target=self.print, args=(data, step))
        job.start()

    def print(self, data: Dict[str, torch.Tensor], step):
        modals = data.keys()
        arrays = zip(*[torch.split(x, 1, dim=0) for x in data.values()])

        for images in arrays:
            fig, axes = plt.subplots(1, 4)
            view = [(modal, image) for modal, image in zip(
                modals, images) if modal in ['t1', 't2', 't1ce', 'pred']]
            mark = {key: label.item() for key, label in zip(
                modals, images) if key in ['number', 'layer']}
            for i, (modal, image) in enumerate(view):
                image = image.squeeze(dim=0).permute(1, 2, 0).numpy()
                ax = axes[i]
                ax.set_title(
                    modal + f"#{mark['number']}, {mark['layer']}")
                ax.axis('off')
                ax.imshow(image, cmap='gray')
            fig.tight_layout()
            plt.savefig(f"{self.path}/{self.model_name}/{mark['number']}-{mark['layer']}.png",
                        bbox_inches='tight', dpi=500)
            plt.close(fig)
            plt.clf()
