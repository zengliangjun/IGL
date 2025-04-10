import torch
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from hydra.utils import instantiate, get_class

# Base class for RL tasks
class BaseTask():
    def __init__(self, config, device):
        self.config = config
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # self.simulator = instantiate(config=self.config.simulator, device=device)
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: BaseSimulator = SimulatorClass(config=self.config, device=device)
        
        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW

        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt
        self.num_envs = self.config.num_envs

        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)

        self._setup_manager()
        # create envs, sim and viewer
        self._load_assets()
        self._get_env_origins()
        self._create_envs()        
        # self._create_sim()
        self.simulator.prepare_sim()
        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if self.headless == False:
            self.debug_viz = False
            self.simulator.setup_viewer()
            ###########################################################################
            # Jiawei: Should be removed
            ###########################################################################
            self.viewer = self.simulator.viewer

        self._init()
        self._post_init()

        ###########################################################################
        #### Jiawei: Should be removed
        ###########################################################################
        # self.gym = self.simulator.gym
        # self.sim = self.simulator.sim
        if self.headless == False:
            self.viewer = self.simulator.viewer

    def _init(self):
        for _key in self.managers:
            self.managers[_key].init()

    def _post_init(self):
        for _key in self.managers:
            self.managers[_key].post_init()

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def reset_all(self):
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        assert hasattr(self, "actions_manager")
        actor_state = self.actions_manager.zeros()
        items = self.step(actor_state)
        assert ('obs_buf_dict' in items)
        return items['obs_buf_dict']

    # def _refresh_env_idx_tensors(self, env_ids):
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                 gymtorch.unwrap_tensor(self.all_root_states),
    #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                             gymtorch.unwrap_tensor(self.dof_state),
    #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions
    ###########################################################################
    def _get_env_origins(self):
        self.terrain_manager.pre_init()

    def _load_assets(self):
        self.simulator.load_assets()
        self.robotdata_manager.pre_init()

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # env_config = self.config
        self.simulator.create_envs(self.num_envs, 
                                    self.terrain_manager.env_origins, 
                                    self.robotdata_manager.base_init_state)
    @property
    def namespace(self):
        from humanoidverse.envs.base_task.term import register
        return register.current_namespace

    def _setup_manager(self):
        self.managers = {}
        from humanoidverse.envs.base_task.term import register
        _registry = register.registry[self.namespace]
        for _name in _registry:
            _manager = _registry[_name](self)
            self.managers[_name] = _manager
            setattr(self, _name, _manager)

