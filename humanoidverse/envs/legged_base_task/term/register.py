from humanoidverse.envs.base_task.term import register
import copy

core_registry = copy.deepcopy(register.registry[register.core_namespace]) #copy.deepcopy(register.baseregistry)

# core

trainer_registry = copy.deepcopy(core_registry)
evaluater_registry = copy.deepcopy(core_registry)
# evaluater

# level 2

core_namespace: str  = "legged_core_task"
trainer_namespace: str  = "legged_trainer_task"
evaluater_namespace: str  = "legged_evaluater_task"
register.registry[core_namespace] = core_registry
register.registry[trainer_namespace] = trainer_registry
register.registry[evaluater_namespace] = evaluater_registry

############ REWARDS ############
legged_base_rewards_registry = copy.deepcopy(register.rewards_registry[register.core_namespace])
from humanoidverse.envs.base_task.rewards import body


register.rewards_registry[trainer_namespace] = legged_base_rewards_registry

