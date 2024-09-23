import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from threading import Lock


class Printer:
    def __init__(self, config):
        self.config = config
        self.num_worker = config.getint("output", "num_worker")
        self.processor = ProcessPoolExecutor(max_workers=self.num_worker)
        self.path = config.get("output", "sample_path")
        self.model_name = config.get("output", "model_name")
        self.lock = Lock()
        self.futures = []

    def __call__(self, *args, **kwargs):
        # future = self.processor.submit(self.print, *args, **kwargs)
        # self.futures.append(future)
        self.print(*args, **kwargs)

    def __del__(self):
        for future in self.futures:
            future.result()

    def print(self, data, step):
        length = len(data)
        if not os.path.exists(f"{self.path}/{self.model_name}"):
            os.makedirs(f"{self.path}/{self.model_name}")
        with self.lock:
            for i, (modal, image) in enumerate(data.items()):
                if len(image.shape) == 4:
                    bs, *_ = image.shape
                    image = image[bs // 2]
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                axes = plt.subplot(1, length, i + 1)
                axes.set_title(modal)
                axes.imshow(image, cmap='gray', vmin=0, vmax=1)
            plt.tight_layout()
            plt.savefig(f"{self.path}/{self.model_name}/{step}.png",
                        bbox_inches='tight', dpi=1600)
            plt.clf()
