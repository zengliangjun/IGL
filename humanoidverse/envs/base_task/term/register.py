from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.base_task.term.statistics import extras
from humanoidverse.envs.base_task.term.foundation import episode, robotdata, terrain
from humanoidverse.envs.base_task.term.mdp import actions, observations, rewards

coreregistry: dict = {} # [str, base.BaseManager] = {}

coreregistry["terrain_manager"] = terrain.BaseTerrainManager         # core
coreregistry["robotdata_manager"] = robotdata.BaseRobotDataManager   # core


core_namespace: str  = "base_task"

registry: dict = {}
registry[core_namespace] = coreregistry

############ REWARDS ############
rewards_registry: dict = {}
