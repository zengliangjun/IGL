from humanoidverse.utils.torch_utils import to_torch, get_axis_params
from humanoidverse.envs.base_task.term import base
import torch

class LeggedContactsManager(base.BaseRobotDataManager):

    def __init__(self, _task):
        super(LeggedContactsManager, self).__init__(_task)


    # stage 1
    def post_init(self):
        robotdata_manager = self.task.robotdata_manager
