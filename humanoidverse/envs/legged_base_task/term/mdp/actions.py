
from humanoidverse.envs.base_task.term.mdp.actions import BaseActionsManager
from humanoidverse.envs.legged_base_task.actions import randomize_ctrl_delay
import torch

class LeggedActionsManager(BaseActionsManager):
    def __init__(self, _task):
        super(LeggedActionsManager, self).__init__(_task)
        self.ctrl_delay = randomize_ctrl_delay.CtrlDelayManager(_task)

    # stage 1
    def init(self):
        self.actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_after_delay = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        ## delay
        self.ctrl_delay.init()

    # stage 2
    def pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)
        ## delay
        self.actions_after_delay = self.ctrl_delay.pre_physics_step(self, actions)


    def post_physics_step(self):
        assert hasattr(self.task, "extras_manager")
        _log_dict = self.extras_manager.log_dict
        clip_action_limit = self.config.robot.control.action_clip_value

        _log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions_after_delay[env_ids] = 0.
        ## delay
        self.ctrl_delay.reset(env_ids)

    def post_step(self):
        self.last_actions[:] = self.actions[:]

    @property
    def actual_actions(self):
        return self.actions_after_delay

    ######################### Observations #########################
    def _get_obs_actions(self,):
        return self.actions
