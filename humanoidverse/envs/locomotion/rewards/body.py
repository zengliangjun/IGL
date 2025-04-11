import torch
from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.torch_utils import quat_rotate_inverse

class UpperBodyRewards(base.BaseManager):
    def __init__(self, _task):
        super(UpperBodyRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity
        robotdata_manager = self.task.robotdata_manager

        torso_ang_vel = quat_rotate_inverse(self.task.simulator._rigid_body_rot[:, robotdata_manager.torso_index],
                                            self.task.simulator._rigid_body_ang_vel[:, robotdata_manager.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    
    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        robotdata_manager = self.task.robotdata_manager
        deviation = torch.abs(self.task.simulator.dof_pos[:, robotdata_manager.upper_dof_indices] - \
                              robotdata_manager.default_dof_pos[:, robotdata_manager.upper_dof_indices])
        return torch.sum(deviation, dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.task.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_penalty_hip_pos(self):
        # TODO HIP is leg ?
        # Penalize the hip joints (only roll and yaw)
        robotdata_manager = self.task.robotdata_manager

        hips_roll_yaw_indices = robotdata_manager.hips_dof_id[1:3] + robotdata_manager.hips_dof_id[4:6]
        hip_pos = self.task.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)
