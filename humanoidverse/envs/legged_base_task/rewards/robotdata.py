import torch
from humanoidverse.envs.base_task.term import base

class RobotDataRewards(base.BaseManager):
    def __init__(self, _task):
        super(RobotDataRewards, self).__init__(_task)

    ######################## LIMITS REWARDS #########################
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.task.simulator.dof_pos - self.task.simulator.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.task.simulator.dof_pos - self.task.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        _robotdata_manager = self.task.robotdata_manager
        _limits = _robotdata_manager.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit
        return torch.sum((torch.abs(self.task.simulator.dof_vel) - _limits).clip(min=0., max=1.), dim=1)
