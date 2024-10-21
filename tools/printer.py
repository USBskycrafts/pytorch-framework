import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from threading import Lock


class Printer:
    def __init__(self, config):
        self.config = config
        self.num_worker = config.getint("output", "num_worker")
        self.processor = ProcessPoolExecutor(max_workers=self.num_worker)
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
            target=self.print, args=(data, step), daemon=True)
        job.start()

    def print(self, data, step):
        fig, axes = plt.subplots(1, 4)
        for i, (modal, image) in enumerate({
            't1': data['t1'],
            't2': data['t2'],
            't1ce': data['t1ce'],
            'pred': data['pred']
        }.items()):
            if len(image.shape) == 4:
                bs, *_ = image.shape
                image = image[bs // 2]
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            ax = axes[i]
            ax.set_title(
                modal + f"#{data['number'][bs // 2]}, {data['layer'][bs // 2]}")
            ax.axis('off')
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        fig.tight_layout()
        plt.savefig(f"{self.path}/{self.model_name}/{data['number'][bs // 2]}-{data['layer'][bs // 2]}.png",
                    bbox_inches='tight', dpi=500)
        plt.close(fig)
        plt.clf()
