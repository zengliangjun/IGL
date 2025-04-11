import torch
from humanoidverse.envs.base_task.term import base

class FeetRewards(base.BaseManager):
    def __init__(self, _task):
        super(FeetRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    #def _reward_penalty_slippage(self):
    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        robotdata_manager = self.task.robotdata_manager
        #feet_manager = self.task.feet_manager

        foot_vel = self.task.simulator._rigid_body_vel[:, robotdata_manager.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * \
                         (torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, :], dim=-1) > 1.), dim=1)

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Jiawei: Key ingredient.
        feet_manager = self.task.feet_manager

        from_air_to_contact = torch.logical_and(feet_manager.contact_filt, ~feet_manager.last_contacts_filt)
        rew_feet_max_height = torch.sum(
            torch.clamp_min(self.config.rewards.desired_feet_max_height_for_this_air - feet_manager.feet_air_max_height, 0) * \
            from_air_to_contact, dim=1) # reward only on first contact with the ground

        return rew_feet_max_height
