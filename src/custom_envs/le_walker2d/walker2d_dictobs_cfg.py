
import numpy as np
from ..le_base import base_practice_cfg as base


class General(base.General):
    EPISODE_TRUNCATION_STEPS_MAX = 5000
    

class PracticeSpace(base.PracticeSpace):
    
    IS_RAND_GOAL_SAMPLING = False
    IS_TERMINATION_IF_OUTSIDE = True
    
    LABELS = np.array([
        'height',   'velocity',  'angle',   'contact_after', 'angle_thigh',  'is_moving_forward',
    ])
    D = np.array([
        [0.8,        -2.0,       -1.0,       1,              -2.0,           1],     # min
        [2.0,        3.0,        1.5,        3,              2.0,            1.1],   # max
        [1.1,        2.0,        0.5,        1,              0,              1],     # mode
        [0.05,       1.0,        0.05,       0,              0.0,            0.5]    # weight [0,1]
    ])[:,[0,1,2,4,5]] # filter
    

class GoalRewardThreshold(base.GoalRewardThreshold):
    IS_ADAPTIVE = False

    MIN = 0.05 * PracticeSpace.RADIUS
    MAX_DEFAULT = 0.2 * PracticeSpace.RADIUS


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = False
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
