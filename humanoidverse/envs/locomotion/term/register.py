from humanoidverse.envs.base_task.term import register as base_register
from humanoidverse.envs.legged_base_task.term import register as legged_register
import copy

trainer_registry = copy.deepcopy(base_register.registry[legged_register.trainer_namespace])
# level 0

# level 1
from humanoidverse.envs.locomotion.term.mdp import command  # mdp
trainer_registry["command_manager"] = command.VelocityCommand

from humanoidverse.envs.locomotion.term.status import feet  # mdp
trainer_registry["feet_manager"] = feet.LeggedFeetManager

############ REWARDS ############
locomotion_rewards_registry = copy.deepcopy(base_register.rewards_registry[legged_register.trainer_namespace])

from humanoidverse.envs.locomotion.rewards import command, feet
locomotion_rewards_registry['command_rewards'] = command.CommandRewards
locomotion_rewards_registry['feet_rewards'] = feet.FeetRewards

####################################
trainer_namespace: str  = "locomotion_trainer_task"
base_register.registry[trainer_namespace] = trainer_registry
base_register.rewards_registry[trainer_namespace] = locomotion_rewards_registry

############    GAIT    ############
gait_rewards_registry = copy.deepcopy(locomotion_rewards_registry)
from humanoidverse.envs.locomotion.rewards import feet_gait
gait_rewards_registry['gait_rewards'] = feet_gait.GaitRewards


gait_trainer_namespace: str  = "gait_trainer_task"
base_register.registry[gait_trainer_namespace] = trainer_registry
base_register.rewards_registry[gait_trainer_namespace] = gait_rewards_registry
