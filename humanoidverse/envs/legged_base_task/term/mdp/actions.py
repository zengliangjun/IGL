
from humanoidverse.envs.base_task.term.mdp.actions import BaseActionsManager
import torch

class LeggedActionsManager(BaseActionsManager):
    def __init__(self, _task):
        super(LeggedActionsManager, self).__init__(_task)

    # stage 1
    def init(self):
        self.actions = torch.zeros(self.task.num_envs, self.dim_actions, dtype=torch.float, device=self.task.device, requires_grad=False)
        self.actions_after_delay = torch.zeros(self.task.num_envs, self.dim_actions, dtype=torch.float, device=self.task.device, requires_grad=False)
        self.last_actions = torch.zeros(self.task.num_envs, self.dim_actions, dtype=torch.float, device=self.task.device, requires_grad=False)

    def post_init(self):
        ######################################### DR related tensors #########################################
        if self.task.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.task.num_envs,
                                            self.task.config.domain_rand.ctrl_delay_step_range[1]+1,
                                            self.dim_actions, dtype=torch.float, device=self.task.device, requires_grad=False)

            self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0],
                                                self.config.domain_rand.ctrl_delay_step_range[1]+1,
                                                (self.num_envs,), device=self.task.device, requires_grad=False)

    # stage 2
    def pre_physics_step(self, actions):
        clip_action_limit = self.task.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.task.device)

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    def post_physics_step(self):
        assert hasattr(self.task, "extras_manager")
        _log_dict = self.task.extras_manager.log_dict
        clip_action_limit = self.task.config.robot.control.action_clip_value

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

        if self.config.domain_rand.randomize_ctrl_delay:
            # self.action_queue[env_ids] = 0.delay:
            self.action_queue[env_ids] *= 0.
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(
                    self.config.domain_rand.ctrl_delay_step_range[0],
                    self.config.domain_rand.ctrl_delay_step_range[1]+1,
                     (len(env_ids),), device=self.device, requires_grad=False)

    def post_step(self):
        self.last_actions[:] = self.actions[:]

    @property
    def actual_actions(self):
        return self.actions_after_delay

    ######################### Observations #########################
    def _get_obs_actions(self,):
        return self.actions
