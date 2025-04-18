import torch
from humanoidverse.envs.legged_base_task.rewards import feet

class FeetRewards(feet.FeetRewards):
    def __init__(self, _task):
        super(FeetRewards, self).__init__(_task)
