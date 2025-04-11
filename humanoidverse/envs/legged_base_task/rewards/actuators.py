import torch
from humanoidverse.envs.base_task.term import base

class ActuatorsRewards(base.BaseManager):
    def __init__(self, _task):
        super(ActuatorsRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_penalty_torques(self):
        # Penalize torques
        _actuators_manager = self.task.actuators_manager
        return torch.sum(torch.square(_actuators_manager.torques), dim=1)


    ######################## LIMITS REWARDS #########################
    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        _actuators_manager = self.task.actuators_manager
        _robotdata_manager = self.task.robotdata_manager
        _limits = _robotdata_manager.torque_limits * self.config.rewards.reward_limit.soft_torque_limit

        return torch.sum((torch.abs(_actuators_manager.torques) - _limits).clip(min=0.), dim=1)
