import torch
from humanoidverse.envs.base_task.term import base
import numpy as np

class BaseEpisode(base.BaseManager):
    def __init__(self, _task):
        super(BaseEpisode, self).__init__(_task)
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.task.dt)

    # stage 1
    def pre_init(self):
        super(BaseEpisode, self).pre_init()
        self._collect_compute_reset()



    def init(self):
        # time out flags
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # termination flags
        self.termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # episode length
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.common_step_counter = torch.tensor(0, device=self.device, dtype=torch.long)

        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl
        # for reward penalty curriculum
        # NOTE it is used after reset
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long) # num_compute_average_epl last termination episode length

    ## stage 2
    def pre_step(self):
        self.time_out_buf[:] = 0
        self.termination_buf[:] = 0

        self.common_step_counter  +=1
        # it will reset to 0
        self.episode_length_buf += 1
        # keep status for later calcute

    def compute_reset(self):
        super(BaseEpisode, self).compute_reset()
        """ Check if environments need to be reset
        """
        # termination
        for _termination in self.check_termination_dict.values():
            _reset = _termination()
            if _reset is None:
                continue
            self.termination_buf |= _reset

        # time_out
        for _time_out in self.check_time_out_dict.values():
            _reset = _time_out()
            if _reset is None:
                continue
            self.time_out_buf |= _reset

        env_ids = self.reset_env_ids
        if len(env_ids) == 0:
            return

        # calcute average_episode_length first
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)
        # reset average_episode_length later
        self.episode_length_buf[env_ids] = 0

    ## help function
    @property
    def reset_buf(self):
        '''
        called only for compute & post_compute & post_step
        '''
        return self.time_out_buf | self.termination_buf

    @property
    def reset_env_ids(self):
        return self.reset_buf.nonzero(as_tuple=False).flatten()

    ## called by agents
    def rand_episode_length(self):
        self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

    def _collect_compute_reset(self):
        self.check_termination_dict = {}
        self.check_time_out_dict = {}

        for _manager in self.task.managers.values():
            _items = dir(_manager)
            for _item in _items:
                if _item.startswith("_check_termination"):
                    _termination_function = getattr(_manager, _item)
                    self.check_termination_dict[_item] = _termination_function
                elif _item.startswith("_check_time_out"):
                    _time_out_function = getattr(_manager, _item)
                    self.check_time_out_dict[_item] = _time_out_function

                #setattr(self, _item, _termination_function)

    ###############################################################################
    def _check_time_out(self):
        return self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
