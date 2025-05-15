from agents.ppo.component import storage
from agents.base_algo import base_algo
from humanoidverse.agents.modules.data_utils_chunk import RolloutStorageForACt
import torch

class Storage(storage.Storage):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Storage, self).__init__(_algo)

    def _preinitobs(self):
        self.storage = RolloutStorageForACt(self.algo)
        ## Register obs keys
        ## please read humanoidverse/utils/helpers.py
        for obs_key, _value in self.algo_obs_dim_dict.items():
            if isinstance(_value, int):
                self.storage.register_key(obs_key, shape=(_value,), dtype = torch.float)
            else:
                _obs_dim, _auxiliary_dims_dict, _auxiliary_dims_length = _value
                self.storage.register_key(obs_key, shape=(_obs_dim,), dtype = torch.float)
                for _key, _dims in _auxiliary_dims_dict.items():
                    _length = _auxiliary_dims_length[_key]
                    self.storage.register_key(f'{obs_key}.{_key}', shape=(_length, _dims), dtype = torch.float)

    ## help functions
    def _step_rollout(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'obs_dict' in _inputs
        assert 'policy_state_dict' in _inputs

        ## Append states to storage
        obs_dict = _inputs['obs_dict']
        for obs_key, _value in obs_dict.items():
            if isinstance(_value, torch.Tensor):
                self.storage.update_key(obs_key, _value)
            else:
                _value, _history_buf_dict = _value
                self.storage.update_key(obs_key, _value)
                for _key, _value in _history_buf_dict.items():
                    self.storage.update_key(f'{obs_key}.{_key}', _value)

        policy_state_dict = _inputs['policy_state_dict']
        for obs_ in policy_state_dict.keys():
            self.storage.update_key(obs_, policy_state_dict[obs_])
