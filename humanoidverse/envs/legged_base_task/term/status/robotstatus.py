from humanoidverse.utils.torch_utils import to_torch, get_axis_params, quat_rotate_inverse
from humanoidverse.utils.spatial_utils.rotations import get_euler_xyz_in_tensor
from humanoidverse.envs.base_task.term import base
import torch

class LeggedStatusManager(base.BaseManager):

    def __init__(self, _task):
        super(LeggedStatusManager, self).__init__(_task)

    # stage 1
    def init(self):
        self.gravity_vec = to_torch(get_axis_params(-1., self.task.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.base_quat = self.task.simulator.base_quat
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.task.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.task.simulator.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.last_dof_vel = torch.zeros_like(self.task.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(self.task.simulator.robot_root_states[:, 7:13])

    # stage 3
    def pre_compute(self):
        # prepare quantities
        self.base_quat[:] = self.task.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.task.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.task.simulator.robot_root_states[:, 10:13])
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.last_dof_vel[env_ids] = 0.

    def post_compute(self):
        self.last_dof_vel[:] = self.task.simulator.dof_vel[:]
        self.last_root_vel[:] = self.task.simulator.robot_root_states[:, 7:13]

    ######################### Observations #########################
    def _get_obs_base_lin_vel(self):
        return self.base_lin_vel

    def _get_obs_base_ang_vel(self):
        return self.base_ang_vel

    def _get_obs_projected_gravity(self,):
        return self.projected_gravity

    def _get_obs_dof_vel(self,):
        return self.task.simulator.dof_vel
