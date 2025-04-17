
from humanoidverse.envs.base_task.term.mdp import rewards

class LeggedRewardsManager(rewards.BaseRewardsManager):
    def __init__(self, _task):
        super(LeggedRewardsManager, self).__init__(_task)
