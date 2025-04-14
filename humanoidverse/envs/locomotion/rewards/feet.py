import torch
from humanoidverse.envs.legged_base_task.rewards import feet
from humanoidverse.utils.torch_utils import quat_rotate_inverse, quat_apply
from isaac_utils.rotations import wrap_to_pi

class FeetRewards(feet.FeetRewards):
    def __init__(self, _task):
        super(FeetRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        feet_manager = self.task.feet_manager
        command_manager = self.task.command_manager

        first_contact = (feet_manager.feet_air_time > self.task.dt) * feet_manager.contact_filt

        rew_airTime = torch.sum((feet_manager.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(command_manager.commands[:, :2], dim=1) > 0.1 #no reward for zero command

        return rew_airTime

    #def _reward_penalty_feet_heading_alignment(self):
    def _reward_feet_heading_alignment(self):
        robotstatus_manager = self.task.robotstatus_manager
        robotdata_manager = self.task.robotdata_manager
        command_manager = self.task.command_manager

        left_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[0]]
        right_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, command_manager.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, command_manager.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(robotstatus_manager.base_quat, command_manager.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))

        return heading_diff_left + heading_diff_right

    ###########################
