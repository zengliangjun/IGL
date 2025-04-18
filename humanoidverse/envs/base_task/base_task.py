import torch
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from hydra.utils import instantiate, get_class
from loguru import logger

# Base class for RL tasks
class BaseTask():
    def __init__(self, config, device):
        ## flags for policy evaluating
        self.config = config
        self.num_envs = self.config.num_envs

        self.debug_viz = False
        self.is_evaluating = False

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        ## set simulator
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: BaseSimulator = SimulatorClass(config=self.config, device=device)

        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW

        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt

        ## set terrain
        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)

        ## instance manager
        self._setup_manager()

        ## load assets
        self.simulator.load_assets()

        ## pre init
        self._pre_init()

        # create envs, sim and viewer
        self._create_envs()

        self.simulator.prepare_sim()
        if self.headless == False:
            self.simulator.setup_viewer()

        ## init
        self._init()

        ## post init
        self._post_init()


    ###########################################################################
    # Helper functions
    def reset_all(self):
        """ Reset all robots"""
        self._reset(torch.arange(self.num_envs, device=self.device))
        self._refresh_envs()

        if hasattr(self, "actions_manager"):
            actor_state = self.actions_manager.zeros()
            items = self.step(actor_state)
            assert ('obs_dict' in items)
            return items['obs_dict']
        else:
            actor_state = {}
            #actions = torch.zeros(self.num_envs, self.robotdata_manager.num_dof, device=self.device, requires_grad=False)
            #actor_state["actions"] = actions
            return self.step(actor_state)

    def rand_episode_length(self):
        if self.is_evaluating:
            return

        if hasattr(self, "episode_manager"):
            self.episode_manager.rand_episode_length()

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True

    # internal functions
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

    def _pre_init(self):
        for _manager in self.managers.values():
            _manager.pre_init()

    def _init(self):
        for _manager in self.managers.values():
            _manager.init()

    def _post_init(self):
        for _manager in self.managers.values():
            _manager.post_init()

    def _setup_manager(self):
        self.managers = {}
        from humanoidverse.envs.base_task.term import register
        _registry = register.registry[self.namespace]
        for _name in _registry:
            _manager = _registry[_name](self)
            self.managers[_name] = _manager
            setattr(self, _name, _manager)

    @property
    def namespace(self):
        from humanoidverse.envs.base_task.term import register
        return register.core_namespace
