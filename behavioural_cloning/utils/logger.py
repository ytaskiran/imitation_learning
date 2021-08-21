import os
from tensorboardX import SummaryWriter
import numpy as np


class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir

        print("///************************///")
        print("Logging outputs to -> ", log_dir)
        print("///************************///")
        self.summary_writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.summary_writer.add_scalar(f"{name}", value, step)
    
    def log_scalars(self, group_name, phase, value_dict, step):
        self.summary_writer.add_scalars(f"{group_name}_{phase}", value_dict, step)

    def log_image(self, name, image, step):
        assert(len(image.shape) == 3)
        self.summary_writer.add_image(f"{name}", image, step)

    def log_video(self, name, video_tensor, step, fps=30):
        assert(len(video_tensor.shape) == 5) 
        self.summary_writer.add_video(f"{name}", video_tensor, step, fps)

    def log_video_rollouts(self, rollouts, step, max_videos=5, fps=30, video_title="Rollouts"):
        videos = [np.transpose(rollout["image_obs"], [0, 1, 3, 2]) for rollout in rollouts]
        max_videos = np.min([max_videos, len(videos)])

        max_length = videos[0].shape[0]
        for i in range(max_videos):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        for i in range(max_videos):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        videos = np.stack(videos[:max_videos], 0)
        self.log_video(video_title, videos, step, fps)

    def log_figure(self, name, phase, figure, step): 
        self.summary_writer.add_figure(f"{name}_{phase}", figure, step)

    def export_scalars(self, log_path=None):
        log_path = os.path.join(self.log_dir, "scalar_data.json") if log_path is None else log_path
        self.summary_writer.export_scalars_to_json(log_path)

    def flush(self):
        self.summary_writer.flush()
