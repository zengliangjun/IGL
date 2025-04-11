import torch
from humanoidverse.envs.base_task.term import base

class StatusRewards(base.BaseManager):
    def __init__(self, _task):
        super(StatusRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        #_robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(self.task.simulator.dof_vel), dim=1)

    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square((_robotstatus_manager.last_dof_vel - self.task.simulator.dof_vel) / self.task.dt), dim=1)

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(_robotstatus_manager.projected_gravity[:, :2]), dim=1)
