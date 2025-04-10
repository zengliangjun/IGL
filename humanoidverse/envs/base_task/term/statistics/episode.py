import torch
from humanoidverse.envs.base_task.term import base
import numpy as np

class BaseEpisode(base.BaseManager):
    def __init__(self, _task):
        super(BaseEpisode, self).__init__(_task)
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.task.dt)

    def init(self):
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

