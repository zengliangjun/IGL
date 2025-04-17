import torch
from humanoidverse.envs.base_task.term import base

class TerminationManager(base.BaseManager):
    def __init__(self, _task):
        super(TerminationManager, self).__init__(_task)

    def _check_termination(self):
        robotdata_manager = self.task.robotdata_manager
        if not hasattr(self.task, 'robotstatus_manager'):
            return None

        robotstatus_manager = self.task.robotstatus_manager
        task = self.task

        _buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.config.termination.terminate_by_contact:

            _buf |= torch.any(torch.norm(task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :], dim=-1) > 1., dim=1)


        if self.config.termination.terminate_by_gravity:
            # print(robotstatus_manager.projected_gravity)
            _buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            _buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)

        if self.config.termination.terminate_by_low_height:
            _buf |= torch.any(task.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                _buf |= out_of_dof_pos_limits > 0.

        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(task.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                _buf |= out_of_dof_vel_limits > 0.

        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                _buf |= out_of_torque_limits > 0.
