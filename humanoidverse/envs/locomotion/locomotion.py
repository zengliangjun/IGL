from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

class LocomotionTrainer(LeggedRobotBase):
    def __init__(self, config, device):
        super(LocomotionTrainer, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.locomotion.term import register
        return register.trainer_namespace

class GaitTrainer(LeggedRobotBase):
    def __init__(self, config, device):
        super(GaitTrainer, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.locomotion.term import register
        return register.gait_trainer_namespace
