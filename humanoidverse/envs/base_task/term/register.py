from humanoidverse.envs.base_task.term.statistics import extras
from humanoidverse.envs.base_task.term.foundation import episode, robotdata, terrain
from humanoidverse.envs.base_task.term.mdp import actions, observations, rewards
from humanoidverse.envs.base_task.terminates import terminations
from humanoidverse.envs.base_task.term.sim2real import push

coreregistry: dict = {} # [str, base.BaseManager] = {}

coreregistry["episode_manager"] = episode.BaseEpisode              # core
coreregistry["terrain_manager"] = terrain.BaseTerrainManager         # core
coreregistry["robotdata_manager"] = robotdata.BaseRobotDataManager   # core
coreregistry["observations_manager"] = observations.BaseObservations
coreregistry["actions_manager"] = actions.ActionsManager

coreregistry["terminations_manager"] = terminations.TerminateManager

coreregistry["rewards_manager"] = rewards.BaseRewardsManager
coreregistry["extras_manager"] = extras.BaseExtrasManager
coreregistry["push_manager"] = push.PushManager

############ REWARDS ############
core_namespace: str  = "base_task"

registry: dict = {}
registry[core_namespace] = coreregistry

############ REWARDS ############
from humanoidverse.envs.base_task.rewards import actions, episode
core_rewards_registry: dict = {}
core_rewards_registry['actions_rewards'] = actions.ActionsRewards
core_rewards_registry['episode_rewards'] = episode.EpisodeRewards

rewards_registry: dict = {}
rewards_registry[core_namespace] = core_rewards_registry