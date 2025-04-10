import numpy as np
import torch
from humanoidverse.envs.base_task.term import base

class BaseTerrainManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseTerrainManager, self).__init__(_task)

    def pre_init(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            # import ipdb; ipdb.set_trace()
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.config.terrain.max_init_terrain_level
            if not self.config.terrain.curriculum: max_init_level = self.config.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.config.terrain.num_rows
            if isinstance(self.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = torch.from_numpy(self.simulator.terrain.env_origins).to(self.device).to(torch.float)
            else:
                self.terrain_origins = self.simulator.terrain.env_origins.to(self.device).to(torch.float)   
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            # import ipdb; ipdb.set_trace()
            # print(self.terrain_origins.shape)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
