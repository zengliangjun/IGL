import torch
from humanoidverse.envs.legged_base_task.term.assistant import robotdata


class LocomotionRobotDataManager(robotdata.LeggedRobotDataManager):
    def __init__(self, _task):
        super(LocomotionRobotDataManager, self).__init__(_task)

    def post_init(self):
        if self.config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.task.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
