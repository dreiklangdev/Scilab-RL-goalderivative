
import numpy as np
from ..le_base import base_practice_cfg as base


class General(base.General):
    IS_OBSERVATION_GOAL_EXTENDED = True


class PracticeSpace(base.PracticeSpace):
    IS_TERMINATION_IF_OUTSIDE = True
    REWARD_IF_OUTSIDE = 0

    LABELS = np.array([
        'height',   'velocity'
    ])
    D = np.array([
        [1.0,        -2.0],  # min
        [2.0,        3.0],   # max
        [1.3,        -1.0],  # mode
        [0.05,       1.0]    # weight [0,1]
    ])

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        STRAT = base.PracticeSpace.RandomGoalSampling.Strat.GENERALIST


class GoalRewardThreshold(base.GoalRewardThreshold):
    pass


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = True
