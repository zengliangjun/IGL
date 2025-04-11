from humanoidverse.envs.base_task.term import base

class LeggedContactsManager(base.BaseManager):

    def __init__(self, _task):
        super(LeggedContactsManager, self).__init__(_task)


    # stage 1
    #def post_init(self):
    #    robotdata_manager = self.task.robotdata_manager
