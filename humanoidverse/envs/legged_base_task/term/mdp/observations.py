from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.base_task.term.mdp import observations
import copy
import torch

class LeggedObservations(observations.BaseObservations):
    def __init__(self, _task):
        super(LeggedObservations, self).__init__(_task)
        self.noise_scales = copy.deepcopy(self.config.obs.noise_scales)

    # stage 1
    def init(self):
        super(LeggedObservations, self).init()
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}

    def post_init(self):
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}

        self._collect_observationss()

    # stage 3
    def compute(self):
        """ Computes observations
        """
        # compute Algo observations
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = dict()
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.config.obs.obs_scales, self.noise_scales)

        history_manager = self.task.history_manager

        # Compute history observations
        history_obs_list = history_manager.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.config.obs.obs_scales, self.noise_scales)

        self._post_config_observation_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)

    def _collect_observationss(self):
        self.observationss_dict = {}
        for _key in self.task.managers:
            _manager = self.task.managers[_key]
            _items = dir(_manager)
            for _item in _items:
                if _item.startswith("_get_obs_"):
                    _obs_function = getattr(_manager, _item)
                    setattr(self, _item, _obs_function)
