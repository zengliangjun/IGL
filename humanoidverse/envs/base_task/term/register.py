from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.base_task.term.statistics import extras
from humanoidverse.envs.base_task.term.foundation import episode, robotdata, terrain
from humanoidverse.envs.base_task.term.mdp import actions, observations, rewards

from humanoidverse.envs.base_task.terminates import terminations


coreregistry: dict = {} # [str, base.BaseManager] = {}

coreregistry["terrain_manager"] = terrain.BaseTerrainManager         # core
coreregistry["robotdata_manager"] = robotdata.BaseRobotDataManager   # core
coreregistry["observations_manager"] = observations.BaseObservations
coreregistry["rewards_manager"] = rewards.BaseRewardsManager
coreregistry["episode_manager"] = episode.BaseEpisode              # core

coreregistry["terminations_manager"] = terminations.TerminateManager

core_namespace: str  = "base_task"

registry: dict = {}
registry[core_namespace] = coreregistry

############ REWARDS ############
from humanoidverse.envs.base_task.rewards import actions
core_rewards_registry: dict = {}
core_rewards_registry['actions_rewards'] = actions.ActionsRewards

rewards_registry: dict = {}
rewards_registry[core_namespace] = core_rewards_registry