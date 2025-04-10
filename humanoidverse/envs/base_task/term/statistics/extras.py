from humanoidverse.envs.base_task.term import base

class BaseExtrasManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseExtrasManager, self).__init__(_task)

    def init(self):
        self.extras = {}
        self.log_dict = {}

