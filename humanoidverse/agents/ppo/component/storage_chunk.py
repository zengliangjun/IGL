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
        for obs_key, _value in self.algo.env.config.robot.algo_obs_dim_dict.items():
            if isinstance(_value, int):
                self.storage.register_key(obs_key, shape=(_value,), dtype = torch.float)
            else:
                _obs_dim, _auxiliary_dims_dict, _auxiliary_dims_length = _value
                self.storage.register_key(obs_key, shape=(_obs_dim,), dtype = torch.float)
                for _key, _dims in _auxiliary_dims_dict.items():
                    _length = _auxiliary_dims_length[_key]
                    self.storage.register_key(f'{obs_key}.{_key}', shape=(_length, _dims + 1), dtype = torch.float)

    # stage 1
    def pre_init(self):
        super(Storage, self).pre_init()
        # record chunk_actions
        _chunk_size = self.algo.config.module_dict.actor.layer_config.chunk_size
        self.storage.register_key('chunk_actions', shape=(_chunk_size, self.num_act), dtype=torch.float)


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

    def _compute_chunk_rewards(self):
        dones = self.storage.query_key('dones')                      # steps, b, 1
        rewards = self.storage.query_key('rewards')                  # steps, b, 1
        actions = self.storage.query_key('actions')                  # steps, b, dims
        chunk_actions = self.storage.query_key('chunk_actions')      # steps, b, _chunk, dims

        _chunk_valid_mask = torch.zeros(chunk_actions.shape[1: 3], dtype = torch.bool, device = actions.device)  # b, _chunk
        _rela_chunk_actions = torch.zeros(chunk_actions.shape[1: ], device = actions.device)  # b, _chunk, dims

        num_steps = chunk_actions.shape[0]
        for step in reversed(range(num_steps)):
            if step != num_steps - 1:
                _current_chunk = chunk_actions[step]
                _diff = torch.abs(_rela_chunk_actions - _current_chunk)            # b, _chunk, dims
                _diff = torch.mean(_diff, dim = -1)                                # b, _chunk
                _diff = torch.sum(_diff * _chunk_valid_mask.float(), dim = -1, keepdim= True)
                _valid_count = torch.sum(_chunk_valid_mask.float(), dim = -1, keepdim= True)
                _isnan_mask = _valid_count == 0
                _diff /= _valid_count
                _diff[_isnan_mask] = 0
                _diff_reward = torch.exp(-_diff) * 0.05
                _diff_reward[_isnan_mask] = 0
                rewards[step: step+1] += _diff_reward

            # copy data
            _current_dones = dones[step]    # b, 1
            _chunk_valid_mask[_current_dones[:, 0]] = False

            _chunk_valid_mask[:, 1:] = _chunk_valid_mask[:, :-1]                # b, _chunk
            _chunk_valid_mask[:, :1] = torch.logical_not(_current_dones)

            _current_actions = actions[step]                                     # b, dims
            _rela_chunk_actions[:, 1:] = _rela_chunk_actions[:, :-1]               # b, _chunk, dims
            _rela_chunk_actions[:, :1] = _current_actions[:, None, :]

    def _compute(self, _inputs):
        ###
        self._compute_chunk_rewards()
        return super(Storage, self)._compute(_inputs)
