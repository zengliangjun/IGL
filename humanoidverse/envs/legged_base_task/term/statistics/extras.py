import torch
from humanoidverse.envs.base_task.term.statistics import extras

class LeggedExtrasManager(extras.BaseExtrasManager):
    def __init__(self, _task):
        super(LeggedExtrasManager, self).__init__(_task)

    # stage 1
    def init(self):
        super(LeggedExtrasManager, self).init()
        #self.extras = {}
        #self.log_dict = {}

    # stage 3
    def post_compute(self):
        self.extras["to_log"] = self.log_dict
        if not hasattr(self.task, "episode_manager"):
            return
        episode_manager = self.task.episode_manager
        self.extras["time_outs"] = episode_manager.time_out_buf

