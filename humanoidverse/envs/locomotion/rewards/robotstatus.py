import torch
from humanoidverse.envs.legged_base_task.rewards import robotstatus

class StatusRewards(robotstatus.StatusRewards):
    def __init__(self, _task):
        super(StatusRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.square(_robotstatus_manager.base_lin_vel[:, 2])

    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(_robotstatus_manager.base_ang_vel[:, :2]), dim=1)
