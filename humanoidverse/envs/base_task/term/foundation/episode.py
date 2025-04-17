import torch
from humanoidverse.envs.base_task.term import base
import numpy as np

class BaseEpisode(base.BaseManager):
    def __init__(self, _task):
        super(BaseEpisode, self).__init__(_task)
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.task.dt)

    def init(self):
        # time out flags
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # termination flags
        self.termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # episode length
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    ## stage 2
    def pre_step(self):
        self.time_out_buf[:] = 0
        self.termination_buf[:] = 0



    ## help function
    @property
    def reset_buf(self):
        '''
        called only for compute & post_compute & post_step
        '''
        return self.time_out_buf | self.termination_buf

    ## called by agents
    def rand_episode_length(self):
        self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
