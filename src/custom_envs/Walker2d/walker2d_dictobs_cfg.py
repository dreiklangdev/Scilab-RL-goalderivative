
from enum import Enum
import numpy as np

EPISODE_TRUNCATION_STEPS_MAX = 1000


class PracticeSpace:
    # every training must be inside practice space (reach, hold, recover, etc.)
    # what if contradictory goals?
    # practice/train space
    # = same as all(!) reasonable/possible/unconstrained states (incl. start-state) (transfer-learning?)
    # TODO how to find perfect practice space? (watch/oversee (at the start)! eg. too strict vs. too lax, also learn to (slightly) recover?)
    # TODO how to find (near-)impossible goals? (eg. distance vs. velocity)
    # intervals
    #   smaller => faster, focussed
    #   larger => slower, more universal
    # generally: the more goal dims., the better? ("more experienced coach")
    # TODO goal analysis (eg. most failed dim.) on eval
    # sample int (float), if int (float)?
    # TODO only randomize specific dims?
    IS_RAND_GOAL_SAMPLING = True # learn to generalize in (noisy) practice-space
    IS_TERMINATION_IF_OUTSIDE = True # radically decrease state-/searchspace
    REWARD_IF_OUTSIDE = 0

    LABELS = np.array([
        'height',   'velocity',  'angle',   'contact_after', 'angle_thigh',  'is_moving_forward',
    ])
    D = np.array([
        [0.8,        -2.0,       -1.0,       1,              -2.0,           1],     # min
        [2.0,        3.0,        1.5,        3,              2.0,            1.1],   # max
        [1.1,        2.0,        0.5,        1,              0,              1],     # mode
        [0.05,       1.0,        0.05,       0,              0.0,            0.5]    # weight [0,1]
    ])[:,[0,1,2,4,5]] # filter
    DIAMETER = np.linalg.norm(D[1] - D[0])
    DIAMETER_NORMED = np.sqrt(D.shape[1])
    MODE = np.linalg.norm(D[2])
    MODE_RATIO = MODE / DIAMETER
    RADIUS_RATIO = max(MODE_RATIO, 1 - MODE_RATIO)
    RADIUS_NORMED = RADIUS_RATIO * DIAMETER_NORMED


class GoalRewardThreshold:
    # "breadcrumbing"
    # rewards: (sparse > freq.)
    #   too frequent => no movement (idleness, fast-narrow conv.)
    #   too sparse => no improvement (randomness, slow-broad conv.)
    #   too painful => no courage (fearful, no conv.)

    IS_ADAPTIVE = True

    # 0.2
    MIN = 0.0 * PracticeSpace.RADIUS_NORMED # REWARD TOLERANCE
    MAX = 1.0 * PracticeSpace.RADIUS_NORMED # decaying
    # need to earn adaption (only bad performance => no rewards!)

class TrajectoryHalving:
    # further attempts to re-improve current trajectory
    # TODO no halving on truncation?
    IS_ENABLED = True
    class Strat(Enum):
        HALF = 0
        HIGHEST_GOAL_CONVERGENCE = 1
        LOWEST_GOAL_DISTANCE = 2
    STRAT = Strat.LOWEST_GOAL_DISTANCE
