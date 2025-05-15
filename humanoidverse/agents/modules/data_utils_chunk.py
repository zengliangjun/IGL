from humanoidverse.agents.modules.data_utils import RolloutStorage
from agents.base_algo import base_algo
import torch

class RolloutStorageForACt(RolloutStorage):

    def __init__(self, _algo: base_algo.BaseAlgo, device='cpu'):
        super(RolloutStorageForACt, self).__init__(_algo, device)


    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        _buffer_dict = {key: getattr(self, key)[:].flatten(0, 1) for key in self.stored_keys}

        ## caclute chunk
        _buffer_actions = _buffer_dict['actions']
        _buffer_dones = _buffer_dict['dones']

        # calcute max indices for indices
        _envs_idx = indices // self.num_transitions_per_env
        _max_indices = (_envs_idx + 1) * self.num_transitions_per_env

        # init mask done tensor
        _mask_done = torch.zeros((mini_batch_size, ), device = self.device)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start: end]

                _batch_buffer_dict = {key: _buffer_dict[key][batch_idx] for key in self.stored_keys}

                ## caclute chunk
                batch_max_indices = _max_indices[start: end]

                _chunk_size = self.config.module_dict.actor.layer_config.chunk_size
                _actions_steps = []
                _dones_steps = []

                _mask_done[:] = 0
                for _steps in range(1, _chunk_size + 1):
                    _steps_idx = batch_idx + _steps

                    # check idx over num_transitions_per_env
                    _steps_done = _steps_idx >= batch_max_indices
                    _steps_idx[_steps_done] = 0
                    _mask_done += _steps_done.float()

                    # dones flags
                    _step_batch_dones = _buffer_dones[_steps_idx].clone()
                    _mask_done += _step_batch_dones[:, 0].float()
                    _step_batch_dones[:, 0] = _mask_done > 0

                    # actions
                    _step_batch_actions = _buffer_actions[_steps_idx].clone()
                    _done_flags = _mask_done > 0
                    _step_batch_actions[_done_flags] = 0

                    _actions_steps.append(_step_batch_actions)

                    print("step: ", _steps, torch.sum(_step_batch_dones.float()), torch.sum(_steps_done.float()))
                    _dones_steps.append(_step_batch_dones)

                _actions_steps = torch.stack(_actions_steps, dim = 1)
                _dones_steps = torch.cat(_dones_steps, dim = -1)
                _batch_buffer_dict['chunk_actions'] = _actions_steps   # b * chunk * dims
                _batch_buffer_dict['chunk_dones'] = _dones_steps       # b * chunk

                yield _batch_buffer_dict
                # ['actor_obs', 'critic_obs', 'actions', 'rewards', 'dones', 'values', 'returns', 'advantages', 'actions_log_prob', 'action_mean', 'action_sigma']


