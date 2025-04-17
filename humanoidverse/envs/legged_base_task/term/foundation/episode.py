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


    def compute_reset(self):
        super(LeggedEpisode, self).compute_reset()
        """ Check if environments need to be reset
        """
        self._check_termination()
        self._check_time_out()

        env_ids = self.reset_env_ids
        if len(env_ids) == 0:
            return
        self.episode_length_buf[env_ids] = 0

        # for curriculum
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def reset(self, env_ids):
        pass

    def post_step(self):
        self.last_episode_length_buf[:] = self.episode_length_buf[:]

    @property
    def reset_env_ids(self):
        return self.reset_buf.nonzero(as_tuple=False).flatten()

    def _check_termination(self):
        robotdata_manager = self.task.robotdata_manager
        if not hasattr(self.task, 'robotstatus_manager'):
            return

        robotstatus_manager = self.task.robotstatus_manager
        task = self.task

        if self.config.termination.terminate_by_contact:

            self.termination_buf |= torch.any(torch.norm(task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :], dim=-1) > 1., dim=1)


        if self.config.termination.terminate_by_gravity:
            # print(robotstatus_manager.projected_gravity)
            self.termination_buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            self.termination_buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)

        if self.config.termination.terminate_by_low_height:
            self.termination_buf |= torch.any(task.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                self.termination_buf |= out_of_dof_pos_limits > 0.

        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(task.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                self.termination_buf |= out_of_dof_vel_limits > 0.

        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                self.termination_buf |= out_of_torque_limits > 0.

    def _check_time_out(self):
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
