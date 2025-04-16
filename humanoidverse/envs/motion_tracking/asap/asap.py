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

    ## help function
    def next_task(self):
        assert hasattr(self, "robotdata_manager")
        self.robotdata_manager.next_task()
        super(ASAPMotionEvaluater, self).reset_all()

    def set_is_evaluating(self):
        super(ASAPMotionEvaluater, self).set_is_evaluating()
        assert hasattr(self, "robotdata_manager")

        ## update motion data random sample
        self.robotdata_manager.with_evaluating()

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.evaluater_namespace


class ASAPMotionPlayer(ASAPMotionEvaluater):
    def __init__(self, config, device):
        super(ASAPMotionPlayer, self).__init__(config, device)
        self.is_motion_player = True

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.player_namespace
