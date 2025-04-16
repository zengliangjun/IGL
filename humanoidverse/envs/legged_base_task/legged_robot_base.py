import torch
from humanoidverse.envs.base_task.base_task import BaseTask

class LeggedRobotBase(BaseTask):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True

    def _init(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init()
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)


    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if 'actions' in actor_state:
            actions = actor_state["actions"]
            # stage 2
            self._pre_physics_step(actions)
            self._physics_step()
            self._post_physics_step()
        else:
            self.render()
            for _ in range(self.config.simulator.config.sim.control_decimation):
                self.simulator.simulate_at_each_physics_step()

        # stage 3
        self._compute3()

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

        return items

    # stage 2
    def _pre_physics_step(self, actions):
        if not hasattr(self, "actions_manager"):
            return

        self.actions_manager.pre_physics_step(actions)

    def _physics_step(self):
        self.render()
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

    # stage 3
    def _compute3(self):
        self._refresh_sim_tensors()

        ## 3.1
        self._pre_compute()
        ## check terminations
        assert hasattr(self, "episode_manager")
        self.episode_manager.check_termination()
        ## compute reward
        if hasattr(self, "rewards_manager"):
            self.rewards_manager.compute_reward()
        ## reset terminations
        self._reset()

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self._post_compute()

    def _pre_compute(self):
        for _key in self.managers:
            self.managers[_key].pre_compute()

    def _reset(self, _env_ids = None):
        assert hasattr(self, "episode_manager")
        if _env_ids is None:
            _env_ids = self.episode_manager.reset_env_ids
        if 0 == len(_env_ids):
            return

        self.need_to_refresh_envs[_env_ids] = True
        for _key in self.managers:
            self.managers[_key].reset(_env_ids)

    def _compute(self):
        for _key in self.managers:
            self.managers[_key].compute()

    def _post_compute(self):
        for _key in self.managers:
            self.managers[_key].post_compute()

    # stage 4
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if not self.viewer or not self.debug_viz:
            return

        # draw height lines
        self.gym.clear_lines(self.viewer)
        self._refresh_sim_tensors()
        for _key in self.managers:
            self.managers[_key].draw_debug_vis()

    @property
    def namespace(self):
        from humanoidverse.envs.legged_base_task.term import register
        return register.core_namespace

    def _plot_domain_rand_params(self):
        raise NotImplementedError

    def get_mppi_buffers(self, env_ids):
        """ Get buffers for MPPI
        MPPI algo need to store the buffers to replicate environments
        """
        '''
        return {
            "dof_pos": copy.deepcopy(self.simulator.dof_pos[env_ids]),
            "dof_vel": copy.deepcopy(self.simulator.dof_vel[env_ids]),
            "base_quat": copy.deepcopy(self.base_quat[env_ids]),
            "base_lin_vel": copy.deepcopy(self.base_lin_vel[env_ids]),
            "base_ang_vel": copy.deepcopy(self.base_ang_vel[env_ids]),
            "projected_gravity": copy.deepcopy(self.projected_gravity[env_ids]),
            "torques": copy.deepcopy(self.torques[env_ids]),
            "actions": copy.deepcopy(self.actions[env_ids]),
            "last_actions": copy.deepcopy(self.last_actions[env_ids]),
            "last_dof_vel": copy.deepcopy(self.last_dof_vel[env_ids]),
            "episode_length_buf": copy.deepcopy(self.episode_length_buf[env_ids]),
            "reset_buf": copy.deepcopy(self.reset_buf[env_ids]),
            "time_out_buf": copy.deepcopy(self.time_out_buf[env_ids]),
            "feet_air_time": copy.deepcopy(self.feet_air_time[env_ids]),
            "last_contacts": copy.deepcopy(self.last_contacts[env_ids]),
            "last_contacts_filt": copy.deepcopy(self.last_contacts_filt[env_ids]),
            "feet_air_max_height": copy.deepcopy(self.feet_air_max_height[env_ids]),
        }
        '''
        pass
