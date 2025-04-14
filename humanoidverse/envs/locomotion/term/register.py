from humanoidverse.envs.base_task.term import register as base_register
from humanoidverse.envs.legged_base_task.rewards import body
from humanoidverse.envs.legged_base_task.term import register as legged_register
import copy

locomotion_registry = copy.deepcopy(base_register.registry[legged_register.current_namespace])
# level 0


# level 1
from humanoidverse.envs.locomotion.term.mdp import command  # mdp
locomotion_registry["command_manager"] = command.VelocityCommand

from humanoidverse.envs.locomotion.term.status import feet  # mdp
locomotion_registry["feet_manager"] = feet.LeggedFeetManager

######
current_namespace: str  = "locomotion_task"
base_register.registry[current_namespace] = locomotion_registry

############ REWARDS ############
locomotion_rewards_registry = copy.deepcopy(base_register.rewards_registry[legged_register.current_namespace])

from humanoidverse.envs.locomotion.rewards import command, feet
locomotion_rewards_registry['command_rewards'] = command.CommandRewards
locomotion_rewards_registry['feet_rewards'] = feet.FeetRewards

base_register.rewards_registry[current_namespace] = locomotion_rewards_registry

