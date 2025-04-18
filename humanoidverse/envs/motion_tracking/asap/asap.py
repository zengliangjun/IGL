from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

class ASAPMotionTrainer(LeggedRobotBase):
    def __init__(self, config, device):
        super(ASAPMotionTrainer, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.trainer_namespace

class ASAPMotionEvaluater(LeggedRobotBase):
    def __init__(self, config, device):
        super(ASAPMotionEvaluater, self).__init__(config, device)

    def next_task(self):
        self.robotdata_manager.next_task()

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.evaluater_namespace

class ASAPMotionPlayer(ASAPMotionEvaluater):
    def __init__(self, config, device):
        super(ASAPMotionPlayer, self).__init__(config, device)

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.player_namespace
