
class BaseManager():
    def __init__(self, _task):
        self.task = _task
        self.num_envs = _task.num_envs
        self.device = _task.device
        self.config = _task.config

    # stage 1
    def pre_init(self):
        # init robot data
        # init terrain
        pass

    def init(self):
        # init buffer data
        pass

    #def init_domain_rand(self):
    def post_init(self):
        # init domain rand data
        pass

    # stage 2
    def pre_physics_step(self, actions):
        # actions, actuators
        pass

    def physics_step(self):
        pass

    def post_physics_step(self):
        pass

    # stage 3
    # compute
    ## post_physics_step
    #def pre_compute_observations(self):
    def pre_compute(self):
        pass

    def check_termination(self):
        # only for episode
        # update to compute
        pass

    def compute_reward(self):
        # only for ewards
        # update to compute
        pass

    def reset(self, env_ids):
        pass

    #def compute_observations(self):
    def compute(self):
        pass

    #def post_compute_observations(self):
    def post_compute(self):
        pass

    # stage 4
    def draw_debug_vis(self):
        pass
