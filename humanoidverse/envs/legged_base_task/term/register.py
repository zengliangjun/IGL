from humanoidverse.envs.base_task.term import register
import copy

core_registry = copy.deepcopy(register.registry[register.core_namespace]) #copy.deepcopy(register.baseregistry)

# core
from humanoidverse.envs.legged_base_task.term.foundation import actuators, robotdata
core_registry["actuators_manager"] = actuators.LeggedActuatorsManager  # core
core_registry["robotdata_manager"] = robotdata.LeggedRobotDataManager  # core

trainer_registry = copy.deepcopy(core_registry)
# trainer
from humanoidverse.envs.legged_base_task.term.status import feet, robotstatus # status
trainer_registry["feet_manager"] = feet.LeggedFeetManager
trainer_registry["robotstatus_manager"] = robotstatus.LeggedStatusManager

from humanoidverse.envs.legged_base_task.term.assistant import history # assistant
trainer_registry["history_manager"] = history.LeggedHistoryManager


from humanoidverse.envs.legged_base_task.term.mdp import actions, \
    command                   # mdp
trainer_registry["command_manager"] = command.LeggedCommandManager
trainer_registry["actions_manager"] = actions.LeggedActionsManager

# level 2
from humanoidverse.envs.legged_base_task.term.sim2real import push
trainer_registry["push_manager"] = push.LeggedPushManager


from humanoidverse.envs.legged_base_task.term.currculum import observations_noise_currculum, \
                                                               reward_limits_curriculum
trainer_registry["observations_noise_currculum"] = observations_noise_currculum.NoiseCurrculum
trainer_registry["reward_limits_curriculum"] = reward_limits_curriculum.LimitsCurrculum

# level 3

evaluater_registry = copy.deepcopy(core_registry)
# evaluater
evaluater_registry["robotstatus_manager"] = robotstatus.LeggedStatusManager
evaluater_registry["history_manager"] = history.LeggedHistoryManager

evaluater_registry["command_manager"] = command.LeggedCommandManager
evaluater_registry["actions_manager"] = actions.LeggedActionsManager
# level 2

core_namespace: str  = "legged_core_task"
trainer_namespace: str  = "legged_trainer_task"
evaluater_namespace: str  = "legged_evaluater_task"
register.registry[core_namespace] = core_registry
register.registry[trainer_namespace] = trainer_registry
register.registry[evaluater_namespace] = evaluater_registry

############ REWARDS ############
legged_base_rewards_registry = copy.deepcopy(register.rewards_registry[register.core_namespace])
from humanoidverse.envs.legged_base_task.rewards import actuators, body, feet, robotstatus
legged_base_rewards_registry['actuators_rewards'] = actuators.ActuatorsRewards


legged_base_rewards_registry['robotstatus_rewards'] = robotstatus.StatusRewards
legged_base_rewards_registry['feet_rewards'] = feet.FeetRewards
legged_base_rewards_registry['body_rewards'] = body.UpperBodyRewards

register.rewards_registry[trainer_namespace] = legged_base_rewards_registry

