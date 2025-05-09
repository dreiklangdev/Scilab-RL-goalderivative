
from ..le_base import base_practice_cfg as base


class General(base.General):
    pass

class MetaObservation(base.MetaObservation):
    pass

class PracticeSpace(base.PracticeSpace):
    IS_TERMINATION_IF_OUTSIDE = True
    REWARD_IF_OUTSIDE = 0

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass

class GoalRewardThreshold(base.GoalRewardThreshold):
    MAX_FAC_DEFAULT = 0.3


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = True
