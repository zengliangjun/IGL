from humanoidverse.envs.base_task.term import base
import torch

class LeggedCommandManager(base.BaseRobotDataManager):

    def __init__(self, _task):
        super(LeggedCommandManager, self).__init__(_task)

    # stage 1
    def init(self):
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

    # stage 3
    def pre_compute(self):
        self.command_counter[:] += 1
