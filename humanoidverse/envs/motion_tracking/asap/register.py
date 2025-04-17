from humanoidverse.envs.base_task.term import register as base_register
from humanoidverse.envs.legged_base_task.term import register as legged_register
import copy

# core
core_registry = copy.deepcopy(base_register.registry[legged_register.core_namespace])
# level 0
from humanoidverse.envs.motion_tracking.asap.foundation import asap_robotdata
core_registry["robotdata_manager"] = asap_robotdata.AsapMotion


# trainer
asap_trainer_registry = copy.deepcopy(base_register.registry[legged_register.trainer_namespace])
asap_trainer_registry.update(core_registry)
# level 1
from humanoidverse.envs.motion_tracking.asap.status import asap_robotstatus
asap_trainer_registry["robotstatus_manager"] = asap_robotstatus.AsapStatus
from humanoidverse.envs.motion_tracking.asap.termination import asap_termination
asap_trainer_registry["asap_termination"] = asap_termination.TrackTermination

# level 2
from humanoidverse.envs.motion_tracking.asap.curriculum import terminate_when_motion_far_curriculum
asap_trainer_registry["terminate_when_motion_far_curriculum"] = terminate_when_motion_far_curriculum.MotionfarCurrculum

from humanoidverse.envs.motion_tracking.asap.extends import asap_motion_save
asap_trainer_registry["asap_motion_save"] = asap_motion_save.MotionSave

# evaluater
asap_evaluater_registry = copy.deepcopy(base_register.registry[legged_register.evaluater_namespace])
asap_evaluater_registry.update(core_registry)
asap_evaluater_registry["robotstatus_manager"] = asap_robotstatus.AsapStatus
asap_evaluater_registry["asap_termination"] = asap_termination.TrackTermination


############ REWARDS ############
asap_rewards_registry = copy.deepcopy(base_register.rewards_registry[legged_register.trainer_namespace])

from humanoidverse.envs.motion_tracking.asap.rewards import asap_motionstatus
asap_rewards_registry['robotstatus_rewards'] = asap_motionstatus.ASAPStatusRewards


####################################
player_namespace: str  = "asap_player_task"
base_register.registry[player_namespace] = core_registry

trainer_namespace: str  = "asap_trainer_task"
base_register.registry[trainer_namespace] = asap_trainer_registry
base_register.rewards_registry[trainer_namespace] = asap_rewards_registry

evaluater_namespace: str  = "asap_evaluater_task"
base_register.registry[evaluater_namespace] = asap_evaluater_registry
