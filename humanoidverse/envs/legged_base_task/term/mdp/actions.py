
from humanoidverse.envs.base_task.term.mdp.actions import BaseActionsManager
from humanoidverse.envs.legged_base_task.actions import randomize_ctrl_delay
import torch

class LeggedActionsManager(BaseActionsManager):
    def __init__(self, _task):
        super(LeggedActionsManager, self).__init__(_task)
        self.ctrl_delay = randomize_ctrl_delay.CtrlDelayManager(_task)

    # stage 1
    def init(self):
        super(LeggedActionsManager, self).init()
        self.ctrl_delay.init()

    # stage 2
    def pre_physics_step(self, actions):
        super(LeggedActionsManager, self).pre_physics_step(actions)
        ## delay
        self.compute_actions = self.ctrl_delay.pre_physics_step(self.compute_actions)

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        super(LeggedActionsManager, self).reset(env_ids)
        ## delay
        self.ctrl_delay.reset(env_ids)
