import torch
from humanoidverse.envs.base_task.term.foundation import episode

class LeggedEpisode(episode.BaseEpisode):
    def __init__(self, _task):
        super(LeggedEpisode, self).__init__(_task)

    # stage 1
    def init(self):
        super(LeggedEpisode, self).init()
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl

        self.common_step_counter = torch.tensor(0, device=self.device, dtype=torch.long)
        # for reward penalty curriculum
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long) # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    # stage 3
    def pre_compute(self):
        super(LeggedEpisode, self).pre_compute()
        self.common_step_counter  +=1
        self.episode_length_buf += 1
        self.last_episode_length_buf = self.episode_length_buf.clone()

    def check_termination(self):
        super(LeggedEpisode, self).check_termination()
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

        self.reset_buf |= self.time_out_buf

        env_ids = self.reset_env_ids
        if len(env_ids) == 0:
            return
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        ################ Curriculum #################
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def reset(self, env_ids):
        pass

    @property
    def reset_env_ids(self):
        return self.reset_buf.nonzero(as_tuple=False).flatten()

    def _update_reset_buf(self):
        robotdata_manager = self.task.robotdata_manager
        if not hasattr(self.task, 'robotstatus_manager'):
            return

        robotstatus_manager = self.task.robotstatus_manager
        task = self.task

        if self.config.termination.terminate_by_contact:
            # print("robotdata_manager.termination_contact_indices", robotdata_manager.termination_contact_indices)
            # print("task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :]", task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :])
            # import ipdb; ipdb.set_trace()
            # print("feet contact forces", task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :])
            self.reset_buf |= torch.any(torch.norm(task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :], dim=-1) > 1., dim=1)

            # the name of the contact indiecs can be found by task.simulator.dof_names[robotdata_manager.termination_contact_indices]

            # Step 1: Find which contact indices caused the reset condition
            # exceeding_contact_indices = robotdata_manager.termination_contact_indices[
            #     torch.any(torch.norm(task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :], dim=-1) > 1., dim=0)
            # ]

            # # Step 2: Map these indices to their corresponding names
            # exceeding_contact_names = [task.simulator.body_names[idx] for idx in exceeding_contact_indices]

            # Print or log the names of the contact indices that caused the reset
            # import ipdb; ipdb.set_trace()
            # print("Contact indices causing reset:", exceeding_contact_names)

        # if self.config.termination.terminate_by_lin_vel:
        #     self.reset_buf |= torch.any(torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > self.config.termination_scales.termination_max_base_vel, dim=1)
        # if self.config.termination.terminate_by_ang_vel:
        #     self.reset_buf |= torch.any(torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > self.config.termination_scales.termination_max_base_ang_vel, dim=1)
        if self.config.termination.terminate_by_gravity:
            # print(robotstatus_manager.projected_gravity)
            self.reset_buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            self.reset_buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            self.reset_buf |= torch.any(task.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                self.reset_buf |= out_of_dof_pos_limits > 0.

        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(task.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                self.reset_buf |= out_of_dof_vel_limits > 0.

        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                self.reset_buf |= out_of_torque_limits > 0.

    def _update_timeout_buf(self):
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
