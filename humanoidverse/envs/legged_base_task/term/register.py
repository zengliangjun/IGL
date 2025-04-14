from humanoidverse.envs.base_task.term import register
import copy

legged_base_registry = copy.deepcopy(register.registry[register.current_namespace]) #copy.deepcopy(register.baseregistry)

# level 0
from humanoidverse.envs.legged_base_task.term.foundation import actuators, episode, robotdata
legged_base_registry["actuators_manager"] = actuators.LeggedActuatorsManager  # core
legged_base_registry["episode_manager"] = episode.LeggedEpisode              # core
legged_base_registry["robotdata_manager"] = robotdata.LeggedRobotDataManager  # core

# level 1
from humanoidverse.envs.legged_base_task.term.status import feet, robotstatus # status
legged_base_registry["feet_manager"] = feet.LeggedFeetManager
legged_base_registry["robotstatus_manager"] = robotstatus.LeggedStatusManager

from humanoidverse.envs.legged_base_task.term.assistant import history # assistant
legged_base_registry["history_manager"] = history.LeggedHistoryManager


from humanoidverse.envs.legged_base_task.term.mdp import actions, \
    command, observations, rewards                                     # mdp

legged_base_registry["command_manager"] = command.LeggedCommandManager
legged_base_registry["actions_manager"] = actions.LeggedActionsManager
legged_base_registry["observations_manager"] = observations.LeggedObservations
legged_base_registry["rewards_manager"] = rewards.LeggedRewardsManager

# level 2
from humanoidverse.envs.legged_base_task.term.sim2real import push
legged_base_registry["push_manager"] = push.LeggedPushManager


from humanoidverse.envs.legged_base_task.term.currculum import observations_noise_currculum, \
                                                               reward_limits_curriculum
legged_base_registry["observations_noise_currculum"] = observations_noise_currculum.NoiseCurrculum
legged_base_registry["reward_limits_curriculum"] = reward_limits_curriculum.LimitsCurrculum

# level 3
from humanoidverse.envs.legged_base_task.term.statistics import extras
legged_base_registry["extras_manager"] = extras.LeggedExtrasManager          # core


current_namespace: str  = "legged_base_task"
register.registry[current_namespace] = legged_base_registry

############ REWARDS ############
legged_base_rewards_registry = {}
from humanoidverse.envs.legged_base_task.rewards import actions, actuators, body, episode, feet, robotstatus
legged_base_rewards_registry['actions_rewards'] = actions.ActionsRewards
legged_base_rewards_registry['actuators_rewards'] = actuators.ActuatorsRewards
legged_base_rewards_registry['episode_rewards'] = episode.EpisodeRewards

legged_base_rewards_registry['robotstatus_rewards'] = robotstatus.StatusRewards
legged_base_rewards_registry['feet_rewards'] = feet.FeetRewards
legged_base_rewards_registry['body_rewards'] = body.UpperBodyRewards

register.rewards_registry[current_namespace] = legged_base_rewards_registry

