from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

class ASAPMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        super(ASAPMotionTracking, self).__init__(config, device)

    ## help function
    def next_task(self):
        assert hasattr(self, "robotdata_manager")
        self.robotdata_manager.next_task()
        super(ASAPMotionTracking, self).reset_all()

    @property
    def namespace(self):
        from humanoidverse.envs.motion_tracking.asap import register
        return register.current_namespace
