from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

class ASAPMotionTrainer(LeggedRobotBase):
    def __init__(self, config, device):
        super(ASAPMotionTrainer, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.trainer_namespace

class ASAPMotionPlayer(LeggedRobotBase):
    def __init__(self, config, device):
        super(ASAPMotionPlayer, self).__init__(config, device)

    ## help function
    def next_task(self):
        assert hasattr(self, "robotdata_manager")
        self.robotdata_manager.next_task()
        super(ASAPMotionPlayer, self).reset_all()

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.player_namespace
