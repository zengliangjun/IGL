import torch
from humanoidverse.envs.base_task.term.foundation import robotdata
from humanoidverse.utils.torch_utils import torch_rand_float
from loguru import logger

class LeggedRobotDataManager(robotdata.BaseRobotDataManager):
    def __init__(self, _task):
        super(LeggedRobotDataManager, self).__init__(_task)

    # stage 1
    def init(self):
        super(LeggedRobotDataManager, self).init()

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        ### TODO
        if self.config.robot.get("upper_left_arm_dof_names", None):
            self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
            self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
            self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
            self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]

        if self.config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.task.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
            logger.info("Yuanhang: -1 for the base link (pelvis)")

    # stage 3
    def reset(self, env_ids):
        super(LeggedRobotDataManager, self).reset(env_ids)
        if len(env_ids) == 0:
            return

        self.task.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
        self.task.simulator.dof_vel[env_ids] = 0.

        # base position
        terrain_manager = self.task.terrain_manager
        if terrain_manager.custom_origins:
            self.task.simulator.robot_root_states[env_ids] = self.base_init_state
            self.task.simulator.robot_root_states[env_ids, :3] += terrain_manager.env_origins[env_ids]
            self.task.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
        else:
            self.task.simulator.robot_root_states[env_ids] = self.base_init_state
            self.task.simulator.robot_root_states[env_ids, :3] += terrain_manager.env_origins[env_ids]

        # base velocities
        self.task.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel

