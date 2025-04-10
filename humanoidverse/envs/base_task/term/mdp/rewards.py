import torch
from humanoidverse.envs.base_task.term import base

class BaseRewardsManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseRewardsManager, self).__init__(_task)

    def init(self):
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
