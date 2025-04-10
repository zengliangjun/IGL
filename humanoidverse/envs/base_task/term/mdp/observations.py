from humanoidverse.envs.base_task.term import base

class BaseObservations(base.BaseManager):
    def __init__(self, _task):
        super(BaseObservations, self).__init__(_task)
        self.dim_obs = self.config.robot.policy_obs_dim
        self.dim_critic_obs = self.config.robot.critic_obs_dim

    def init(self):
        self.obs_buf_dict = {}
