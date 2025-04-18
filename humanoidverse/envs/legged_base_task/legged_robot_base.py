import torch
from humanoidverse.envs.base_task.base_task import BaseTask

class LeggedRobotBase(BaseTask):
    def __init__(self, config, device):
        self.init_done = False
        super(LeggedRobotBase, self).__init__(config, device)
        self.init_done = True

    def _init(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super(LeggedRobotBase, self)._init()
        # need _refresh flags
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self._pre_step()

        ## physics_step
        if 'actions' in actor_state:
            actions = actor_state["actions"]
            # stage 2
            self._pre_physics_step(actions)
            self._physics_step()
            self._post_physics_step()
        else:
            self._render()
            for _ in range(self.config.simulator.config.sim.control_decimation):
                self.simulator.simulate_at_each_physics_step()

        self._refresh_sim_tensors()

        ## compute
        self._pre_compute()
        ## check reset flag
        assert hasattr(self, "episode_manager")
        self.episode_manager.compute_reset()
        ## compute reward
        if hasattr(self, "rewards_manager"):
            self.rewards_manager.compute_reward()
        ## reset terminations
        self._reset()

        # set envs
        self._refresh_envs()

        self._compute() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self._post_compute()

        # stage 4
        self._draw_debug_vis()

        items = {}
        if hasattr(self, "observations_manager"):
            items['obs_dict'] = self.observations_manager.obs_buf_dict

        if hasattr(self, "rewards_manager"):
            items['rewards'] = self.rewards_manager.rew_buf

        if hasattr(self, "episode_manager"):
            items['dones'] = self.episode_manager.reset_buf

        if hasattr(self, "extras_manager"):
            items['infos'] = self.extras_manager.extras

        self._post_step()

        return items

    # helper functions
    def _pre_step(self):
        for _manager in self.managers.values():
            _manager.pre_step()

    # physics step
    def _pre_physics_step(self, actions):
        if not hasattr(self, "actions_manager"):
            return
        self.actions_manager.pre_physics_step(actions)

    def _physics_step(self):
        self._render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        if not hasattr(self, "actions_manager"):
            return
        _actions = self.actions_manager.actual_actions
        self.actuators_manager.pre_physics_step(_actions)
        self.simulator.apply_torques_at_dof(self.actuators_manager.torques)

    def _post_physics_step(self):
        pass

    # compute
    def _pre_compute(self):
        for _manager in self.managers.values():
            _manager.pre_compute()

    def _reset(self, _env_ids = None):
        assert hasattr(self, "episode_manager")
        if _env_ids is None:
            _env_ids = self.episode_manager.reset_env_ids
        else:
            # TODO merger _env_ids & self.episode_manager.reset_env_ids
            pass

        if 0 == len(_env_ids):
            return
        # update refresh flags
        self.need_to_refresh_envs[_env_ids] = True
        for _manager in self.managers.values():
            _manager.reset(_env_ids)

    def _compute(self):
        for _manager in self.managers.values():
            _manager.compute()

    def _post_compute(self):
        for _manager in self.managers.values():
            _manager.post_compute()

    # stage 4
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if self.headless:
            return

        if not self.debug_viz:
            return

        self.simulator.clear_lines()
        self._refresh_sim_tensors()
        for _manager in self.managers.values():
            _manager.draw_debug_vis()

    def _post_step(self):
        for _manager in self.managers.values():
            _manager.post_step()

    def _render(self, sync_frame_time=True):
        if self.headless:
            return
        self.simulator.render(sync_frame_time)


    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def _refresh_envs(self):
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False


    @property
    def namespace(self):
        from humanoidverse.envs.legged_base_task.term import register
        return register.core_namespace
