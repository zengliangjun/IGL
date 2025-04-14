import torch
from humanoidverse.envs.legged_base_task.term.status import feet

class LeggedFeetManager(feet.LeggedFeetManager):
    def __init__(self, _task):
        super(LeggedFeetManager, self).__init__(_task)

    # stage 1
    def post_init(self):
        super(LeggedFeetManager, self).post_init()
        # wait for robotdata_manager init
        robotdata_manager = self.task.robotdata_manager

        ###
        self.feet_air_time = torch.zeros(self.num_envs, robotdata_manager.feet_indices.shape[0],
                                         dtype=torch.float, device=self.device, requires_grad=False)

    # stage 3
    def pre_compute(self):
        super(LeggedFeetManager, self).pre_compute()
        self.feet_air_time += self.task.dt  ## TODO update here

    def post_compute(self):
        super(LeggedFeetManager, self).post_compute()
        self.feet_air_time *= ~self.contact_filt

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.feet_air_time[env_ids] = 0.
