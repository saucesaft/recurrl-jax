import os
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np

class TensorBoardLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        # create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(run_dir))
        self.run_dir = run_dir

        print(f"Logging to: {run_dir}")
        print(f"View with: tensorboard --logdir={log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        # log single scalar value
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, value_dict: dict, step: int):
        # log multiple related scalars
        self.writer.add_scalars(tag, value_dict, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        # log distribution of values
        self.writer.add_histogram(tag, values, step)

    def log_video(self, tag: str, video_path: str, step: int, fps: int = 20):
        # tensorboardX doesn't support direct video logging well
        # just log the path as text for now
        self.writer.add_text(f"{tag}/path", str(video_path), step)

    def close(self):
        self.writer.close()
