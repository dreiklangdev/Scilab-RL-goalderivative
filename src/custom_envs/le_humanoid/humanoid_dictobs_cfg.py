
from enum import Enum
import numpy as np
from ..le_base import base_practice_cfg as base


class General(base.General):
    pass


class PracticeSpace(base.PracticeSpace):
    
    IS_RAND_GOAL_SAMPLING = True
    IS_TERMINATION_IF_OUTSIDE = True
    REWARD_IF_OUTSIDE = 0

    LABELS = np.array([
        'height',   'velocity',  'angle',   'contact_after', 'angle_thigh',  'is_moving_forward',
    ])
    D = np.array([
        [0.8,        -2.0],  # min
        [2.0,        3.0],   # max
        [1.1,        1.0],   # mode
        [0.05,       1.0]    # weight [0,1]
    ])


class GoalRewardThreshold(base.GoalRewardThreshold):

    IS_ADAPTIVE = False

    MIN = 0.05 * PracticeSpace.RADIUS
    MAX_DEFAULT = 0.2 * PracticeSpace.RADIUS


class TrajectoryHalving(base.TrajectoryHalving):

    IS_ENABLED = True
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
