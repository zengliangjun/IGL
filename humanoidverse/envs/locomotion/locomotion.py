from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

class BaseLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        super(BaseLocomotion, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.locomotion.term import register
        return register.current_namespace

class GaitLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        super(GaitLocomotion, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.locomotion.term import register
        return register.gait_namespace
