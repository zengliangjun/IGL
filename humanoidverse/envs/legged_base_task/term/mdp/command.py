from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.torch_utils import to_torch
import torch

class LeggedCommandManager(base.BaseManager):

    def __init__(self, _task):
        super(LeggedCommandManager, self).__init__(_task)

    # stage 1
    def init(self):
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        # sample as rewards
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

    # stage 3
    def pre_compute(self):
        self.command_counter[:] += 1
