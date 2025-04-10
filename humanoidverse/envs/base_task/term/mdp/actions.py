import torch
from humanoidverse.envs.base_task.term import base

class BaseActionsManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseActionsManager, self).__init__(_task)
        self.dim_actions = self.task.config.robot.actions_dim

    def zeros(self):
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        return actor_state

    def check(self, num_dof):
        assert num_dof == self.dim_actions, "Number of DOFs must be equal to number of actions"
