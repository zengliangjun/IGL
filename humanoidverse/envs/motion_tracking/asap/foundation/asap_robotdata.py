import torch
import numpy as np
from humanoidverse.envs.legged_base_task.term.foundation import robotdata

from motion_lib import motion_lib_robot
from loguru import logger

def _small_random_quaternions(obj, n, max_angle):
    axis = torch.randn((n, 3), device=obj.device)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
    angles = max_angle * torch.rand((n, 1), device=obj.device)

    # Convert angle-axis to quaternion
    sin_half_angle = torch.sin(angles / 2)
    cos_half_angle = torch.cos(angles / 2)

    q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)
    return q

class AsapMotion(robotdata.LeggedRobotDataManager):
    def __init__(self, _task):
        super(AsapMotion, self).__init__(_task)

    def pre_init(self):
        super(AsapMotion, self).pre_init()
        ## status
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        ## motion
        self.config.robot.motion.step_dt = self.task.dt
        self.motion_lib = motion_lib_robot.MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self.motion_lib.load_motions(random_sample=False)
        self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)

        ## for resample
        self.motion_start_idx = 0
        self.num_motions = self.motion_lib._num_unique_motions
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.task.dt)

    def post_init(self):
        super(AsapMotion, self).post_init()
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]

    def reset(self, env_ids):
        super(AsapMotion, self).reset(env_ids)
        if len(env_ids) != 0:
            if self.task.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
                self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
            else:
                self.motion_start_times[env_ids] = self.motion_lib.sample_time(self.motion_ids[env_ids])

            # only for reset motion res
            _motion_ref = self._motion_ref
            ## stage 2
            self._reset_dofs(env_ids, _motion_ref)
            self._reset_root_states(env_ids, _motion_ref)
        ##
        self.next_motion_ref = self._next_motion_ref

    def post_compute(self):
        super(AsapMotion, self).post_compute()
        self.current_motion_ref = self.next_motion_ref

    def _reset_dofs(self, env_ids, _motion_ref):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        motion_ref = _motion_ref

        dof_pos = motion_ref['dof_pos'][env_ids]
        dof_vel = motion_ref['dof_vel'][env_ids]
        if self.task.is_evaluating:
            self.task.simulator.dof_pos[env_ids] = dof_pos
            self.task.simulator.dof_vel[env_ids] = dof_vel
            return

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        self.task.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.task.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise

    def _reset_root_states(self, env_ids, _motion_ref):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        assert hasattr(self.task, "terrain_manager")
        terrain_manager = self.task.terrain_manager
        motion_ref = _motion_ref

        if terrain_manager.custom_origins:

            self.task.simulator.robot_root_states[env_ids, :3] = motion_ref['root_pos'][env_ids]
            # self.robot_root_states[env_ids, 2] += 0.04 # in case under the terrain
            if self.config.simulator.config.name == 'isaacgym':
                self.task.simulator.robot_root_states[env_ids, 3:7] = motion_ref['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim' or self.config.simulator.config.name == 'isaacsim45':
                from isaac_utils.rotations import xyzw_to_wxyz
                self.task.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_ref['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.task.simulator.robot_root_states[env_ids, 3:7] = motion_ref['root_rot'][env_ids]
                raise NotImplementedError
            self.task.simulator.robot_root_states[env_ids, 7:10] = motion_ref['root_vel'][env_ids]
            self.task.simulator.robot_root_states[env_ids, 10:13] = motion_ref['root_ang_vel'][env_ids]

        else:
            from isaac_utils.rotations import quat_mul, xyzw_to_wxyz

            if self.task.is_evaluating:
                root_pos_noise = 0
                root_rot_noise = 0
                root_vel_noise = 0
                root_ang_vel_noise = 0
            else:
                root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
                root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
                root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
                root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_ref['root_pos'][env_ids]
            root_rot = motion_ref['root_rot'][env_ids]
            root_vel = motion_ref['root_vel'][env_ids]
            root_ang_vel = motion_ref['root_ang_vel'][env_ids]

            self.task.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim' or self.config.simulator.config.name == 'isaacsim45':
                self.task.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            else:
                raise NotImplementedError
            self.task.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.task.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

    @property
    def _motion_ref(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        terrain_manager = self.task.terrain_manager

        offset = terrain_manager.env_origins
        motion_times = (episode_manager.episode_length_buf) * self.task.dt + self.motion_start_times # next frames so +1
        motion_ref = self.motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        return motion_ref

    @property
    def _next_motion_ref(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        terrain_manager = self.task.terrain_manager

        offset = terrain_manager.env_origins
        motion_times = (episode_manager.episode_length_buf + 1) * self.task.dt + self.motion_start_times # next frames so +1
        motion_ref = self.motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        return motion_ref

    @property
    def _next_motion_times(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        return (episode_manager.episode_length_buf + 1) * self.task.dt + self.motion_start_times # next frames so +1
