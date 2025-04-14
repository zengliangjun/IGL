import torch
from humanoidverse.envs.base_task.term import base

class RobotDataRewards(base.BaseManager):
    def __init__(self, _task):
        super(RobotDataRewards, self).__init__(_task)

    ######################## LIMITS REWARDS #########################
