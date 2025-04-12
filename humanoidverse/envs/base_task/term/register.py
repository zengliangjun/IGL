from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.base_task.term.statistics import extras
from humanoidverse.envs.base_task.term.foundation import episode, robotdata, terrain
from humanoidverse.envs.base_task.term.mdp import actions, observations, rewards

baseregistry: dict = {} # [str, base.BaseManager] = {}

baseregistry["terrain_manager"] = terrain.BaseTerrainManager
baseregistry["robotdata_manager"] = robotdata.BaseRobotDataManager


current_namespace: str  = "base_task"

registry: dict = {}
registry[current_namespace] = baseregistry

############ REWARDS ############
rewards_registry: dict = {}
