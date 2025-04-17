import torch
from humanoidverse.utils.torch_utils import to_torch
from humanoidverse.envs.base_task.term import base

class BaseRobotDataManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseRobotDataManager, self).__init__(_task)

    def pre_init(self):
        '''
        simulator had instanced
        '''
        self.num_dof = self.task.simulator.num_dof
        self.num_bodies = self.task.simulator.num_bodies
        self.dof_names = self.task.simulator.dof_names
        self.body_names = self.task.simulator.body_names
        # check dimensions
        if hasattr(self.task, "actions_manager"):
            self.task.actions_manager.check(self.num_dof)

        # other properties
        self.num_bodies = len(self.body_names)
        #self.num_dofs = len(self.dof_names)
        assert len(self.dof_names) == self.num_dof
        base_init_state_list = self.config.robot.init_state.pos + \
                               self.config.robot.init_state.rot + \
                               self.config.robot.init_state.lin_vel + \
                               self.config.robot.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def init(self):
        '''
        envs had created
        '''

        self.dof_pos_limits, \
        self.dof_vel_limits, \
        self.torque_limits = self.task.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()

    def _setup_robot_body_indices(self):
        feet_names = [s for s in self.body_names if self.config.robot.foot_name in s]
        knee_names = [s for s in self.body_names if self.config.robot.knee_name in s]
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.task.simulator.find_rigid_body_indice(feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.task.simulator.find_rigid_body_indice(knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.task.simulator.find_rigid_body_indice(penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.task.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.task.simulator.find_rigid_body_indice(self.torso_name)
