
import numpy as np
from ..le_base import base_practice_cfg as base


class General(base.General):
    EPISODE_TRUNCATION_STEPS_MAX = 1000
    

class PracticeSpace(base.PracticeSpace):    
    IS_TERMINATION_IF_OUTSIDE = True
    
    base.PracticeSpace.init_dims([
        [0.8,        -2.0,          -1.0,       -2.0,           1],     # min
        [2.0,        3.0,           1.5,        2.0,            1.1],   # max
        [1.1,        2.0,           0.5,        0,              1],     # mode
        [0.05,       1.0,           0.05,       0.0,            0.5]    # weight [0,1]
    ],  ['height',   'velocity',    'angle',   'angle_thigh',   'is_moving_forward',])

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        STRAT = base.PracticeSpace.RandomGoalSampling.Strat.INCREMENTALIST


class GoalRewardThreshold(base.GoalRewardThreshold):
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = True
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
