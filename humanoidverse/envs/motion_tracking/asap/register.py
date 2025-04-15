from humanoidverse.envs.base_task.term import register as base_register
from humanoidverse.envs.legged_base_task.rewards import body
from humanoidverse.envs.legged_base_task.term import register as legged_register
import copy

asap_registry = copy.deepcopy(base_register.registry[legged_register.current_namespace])
# level 0
from humanoidverse.envs.motion_tracking.asap.foundation import asap_episode, asap_robotdata
asap_registry["episode_manager"] = asap_episode.ASAPEpisode
asap_registry["robotdata_manager"] = asap_robotdata.AsapMotion

# level 1
from humanoidverse.envs.motion_tracking.asap.status import asap_robotstatus
asap_registry["robotstatus_manager"] = asap_robotstatus.AsapStatus

# level 2
from humanoidverse.envs.motion_tracking.asap.curriculum import terminate_when_motion_far_curriculum
asap_registry["terminate_when_motion_far_curriculum"] = terminate_when_motion_far_curriculum.MotionfarCurrculum

from humanoidverse.envs.motion_tracking.asap.extends import asap_motion_save
asap_registry["asap_motion_save"] = asap_motion_save.MotionSave

############ REWARDS ############
asap_rewards_registry = copy.deepcopy(base_register.rewards_registry[legged_register.current_namespace])

from humanoidverse.envs.motion_tracking.asap.rewards import asap_motionstatus
asap_rewards_registry['robotstatus_rewards'] = asap_motionstatus.ASAPStatusRewards


####################################
current_namespace: str  = "asap_motion_task"
base_register.registry[current_namespace] = asap_registry
base_register.rewards_registry[current_namespace] = asap_rewards_registry
