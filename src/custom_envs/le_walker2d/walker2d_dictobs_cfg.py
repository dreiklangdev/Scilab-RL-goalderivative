
import numpy as np
from ..le_base import base_practice_cfg as base


class General(base.General):
    EPISODE_TRUNCATION_STEPS_MAX = 1000


class MetaObservation(base.MetaObservation):
    IS_ENABLED = True


class PracticeSpace(base.PracticeSpace):    
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True # manually decrease search-/practice-space
    REWARD_ON_TERMINATE = 0
    
    base.PracticeSpace.init_dims([
        [0.8,        -2.0,          -1.0,       -2.0,   ],   # min
        [2.0,        3.0,           1.5,        2.0,    ],   # max
        [1.1,        1.0,           0.5,        0,      ],   # mode
        # [0.05,       1.0,           0.05,       0.0,    ]    # weight [0,1]
        # [0.0,        1.0,            0.0,       0.0,    ]    # weight [0,1]
        [1.0,        1.0,           0.0,        0.0,    ]    # weight [0,1]
    ],  ['height',   'velocity',    'angle',    'angle_thigh'])

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        STRAT = base.PracticeSpace.RandomGoalSampling.Strat.SPECIALIST


class PracticeTime(base.PracticeTime):    
    IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE = True # autom. decrease search-/practice-time
    GRACE_STEPS = 100
    REWARD_ON_TERMINATE = -1


class GoalRewardThreshold(base.GoalRewardThreshold):
    MAX_FRAC_DEFAULT = 0.1
    IS_NUDGING = True
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = True
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
