
from humanoidverse.envs.base_task.term.mdp import observations


class LeggedObservations(observations.BaseObservations):
    def __init__(self, _task):
        super(LeggedObservations, self).__init__(_task)
