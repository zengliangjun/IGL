import torch
from humanoidverse.envs.legged_base_task.term.status import robotstatus
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
)

class AsapStatus(robotstatus.LeggedStatusManager):
    def __init__(self, _task):
        super(AsapStatus, self).__init__(_task)

    # init
    def init(self):
        #_num_bodies = self.num_bodies
        ####
        #self.marker_coords = torch.zeros(self.num_envs,
        #                                _num_bodies,
        #                                3,
        #                                dtype=torch.float,
        #                                device=self.device,
        #                                requires_grad=False) # extend

        #self.ref_body_pos = torch.zeros(self.num_envs, _num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        #
        # self.dif_global_body_pos = torch.zeros(self.num_envs, _num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        #pass
        super(AsapStatus, self).init()

        self.motion_res = None # wait for pre_compute

        # for debug settings TODO check later for must load
        if self.task.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.task.simulator.add_visualize_entities(num_visualize_markers)
        elif self.task.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.task.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass


    @property
    def num_bodies(self):
        robotdata_manager = self.task.robotdata_manager
        _num_bodies = robotdata_manager.num_bodies ## move to robotdata
        if hasattr(robotdata_manager, "num_extend_bodies"):
            _num_bodies += robotdata_manager.num_extend_bodies

        return _num_bodies

    def pre_compute(self):
        super(AsapStatus, self).pre_compute()

        robotdata_manager = self.task.robotdata_manager
        motion_res = robotdata_manager.next_motion_res

        # update motion_res
        self.motion_res = motion_res

        #ref_body_pos = motion_res["rg_pos_t"]
        #self.ref_body_pos[:] = ref_body_pos # for visualization and analysis

        self._pre_rigid_body_compute()


    def post_compute(self):
        if not hasattr(self.task, "extras_manager"):
            return

        robotdata_manager = self.task.robotdata_manager

        ref_body_pos = self.motion_res["rg_pos_t"]
        ref_joint_pos = self.motion_res["dof_pos"] # [num_envs, num_markers, 3]
        ## diff compute - kinematic position
        _dif_global_body_pos = ref_body_pos - self._rigid_body_pos

        upper_body_diff = _dif_global_body_pos[:, robotdata_manager.upper_body_id, :]
        lower_body_diff = _dif_global_body_pos[:, robotdata_manager.lower_body_id, :]
        vr_3point_diff = _dif_global_body_pos[:, robotdata_manager.motion_tracking_id, :]
        joint_pos_diff = ref_joint_pos - self.task.simulator.dof_pos

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        extras_manager = self.task.extras_manager
        extras_manager.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        extras_manager.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        extras_manager.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        extras_manager.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm


    def _pre_rigid_body_compute(self):
        robotdata_manager = self.task.robotdata_manager

        ## rigid_body
        self._rigid_body_pos = self.task.simulator._rigid_body_pos
        self._rigid_body_rot = self.task.simulator._rigid_body_rot
        self._rigid_body_ang_vel = self.task.simulator._rigid_body_ang_vel
        self._rigid_body_vel = self.task.simulator._rigid_body_vel

        ## extend
        if hasattr(robotdata_manager, "extend_body_parent_ids"):

            ################### EXTEND Rigid body POS #####################
            rotated_pos_in_parent = my_quat_rotate(
                self.task.simulator._rigid_body_rot[:, robotdata_manager.extend_body_parent_ids].reshape(-1, 4),
                robotdata_manager.extend_body_pos_in_parent.reshape(-1, 3)
            )
            extend_curr_pos = my_quat_rotate(
                robotdata_manager.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                rotated_pos_in_parent
            ).view(self.num_envs, -1, 3) + self._rigid_body_pos[:, robotdata_manager.extend_body_parent_ids]

            self._rigid_body_pos = torch.cat([self._rigid_body_pos, extend_curr_pos], dim=1)
            ################### EXTEND Rigid body Rotation #####################
            extend_curr_rot = quat_mul(self._rigid_body_rot[:, robotdata_manager.extend_body_parent_ids].reshape(-1, 4),
                                        robotdata_manager.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                        w_last=True).view(self.num_envs, -1, 4)

            self._rigid_body_rot = torch.cat([self._rigid_body_rot, extend_curr_rot], dim=1)
            ################### EXTEND Rigid Body Angular Velocity #####################
            self._rigid_body_ang_vel = torch.cat([self._rigid_body_ang_vel, self._rigid_body_ang_vel[:, robotdata_manager.extend_body_parent_ids]], dim=1)

            ################### EXTEND Rigid Body Linear Velocity #####################
            _rigid_body_ang_vel_global = self.task.simulator._rigid_body_ang_vel[:, robotdata_manager.extend_body_parent_ids]
            angular_velocity_contribution = torch.cross(_rigid_body_ang_vel_global, robotdata_manager.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
            _extend_curr_vel = self._rigid_body_vel[:, robotdata_manager.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
            self._rigid_body_vel = torch.cat([self._rigid_body_vel, _extend_curr_vel], dim=1)

    ######################### Observations #########################
    ## rigid body local pos
    def _get_obs_dif_local_rigid_body_pos(self):
        ref_body_pos = self.motion_res["rg_pos_t"]
        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        #heading_rot = calc_heading_quat(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        #heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        dif_global_body_pos_for_obs_compute = ref_body_pos.view(self.num_envs, -1, 3) - \
                                              self._rigid_body_pos.view(self.num_envs, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), \
                                                 dif_global_body_pos_for_obs_compute.view(-1, 3))

        return dif_local_body_pos_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies*3)

    def _get_obs_local_ref_rigid_body_pos(self):
        ref_body_pos = self.motion_res["rg_pos_t"]

        global_ref_rigid_body_pos = ref_body_pos.view(self.num_envs, -1, 3) - \
            self.task.simulator.robot_root_states[:, :3].view(self.num_envs, 1, 3)  # preserves the body position

        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        return local_ref_rigid_body_pos_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies * 3)

    def _get_obs_vr_3point_pos(self):
        robotdata_manager = self.task.robotdata_manager
        assert hasattr(robotdata_manager, 'motion_tracking_id')
        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)

        ref_body_pos = self.motion_res["rg_pos_t"]
        ref_vr_3point_pos = ref_body_pos.view(self.num_envs, -1, 3)[:, robotdata_manager.motion_tracking_id, :]

        vr_2root_pos = ref_vr_3point_pos - self.task.simulator.robot_root_states[:, 0:3].view((self.num_envs, 1, 3))
        return my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(self.num_envs, -1)

    # rigid body local vel
    def _get_obs_local_ref_rigid_body_vel(self):
        ref_body_vel = self.motion_res["body_vel_t"] # [num_envs, num_markers, 3]

        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        global_ref_body_vel = ref_body_vel.view(self.num_envs, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        return local_ref_rigid_body_vel_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies * 3)

    # phase
    #################### Deepmimic phase ######################
    def _get_obs_ref_motion_phase(self):
        robotdata_manager = self.task.robotdata_manager
        _ref_motion_length = robotdata_manager._motion_lib.get_motion_length(robotdata_manager.motion_ids)
        _ref_motion_phase = robotdata_manager.next_motion_times / _ref_motion_length
        #if not (torch.all(_ref_motion_phase >= 0) and torch.all(_ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
        #    max_phase = _ref_motion_phase.max()
        #    # import ipdb; ipdb.set_trace()
        return _ref_motion_phase.unsqueeze(1)


    def draw_debug_vis(self):
        ref_body_pos = self.motion_res["rg_pos_t"]
        ref_body_pos = ref_body_pos.reshape(self.num_envs, -1, 3)

        for env_id in range(self.num_envs):

            # draw marker joints
            for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                if self.config.robot.motion.visualization.customize_color:
                    _id = pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)
                    color_inner = self.config.robot.motion.visualization.marker_joint_colors[_id]
                else:
                    color_inner = (0.3, 0.3, 0.3)
                color_inner = tuple(color_inner)

                self.task.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)

