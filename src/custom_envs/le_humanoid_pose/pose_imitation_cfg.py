
from ..le_base import base_practice_cfg as base
import numpy as np


class General(base.General):
    LANDMARK_GROUPS = [
    #    [8, 6, 5, 4, 0, 1, 2, 3, 7],   # eyes
    #    [10, 9],                       # mouth
        [11, 13, 15, 17, 19, 15, 21],  # right arm
    #    [11, 23, 25, 27, 29, 31, 27],  # right body side
        [12, 14, 16, 18, 20, 16, 22],  # left arm
    #    [12, 24, 26, 28, 30, 32, 28],  # left body side
    #    [11, 12],                      # shoulder
    #    [23, 24],                      # waist
    ]

    OBSERVATION_DIMS_VISUAL_DETECTION = len(LANDMARK_GROUPS) * len(LANDMARK_GROUPS[0]) * 3
    RENDER_IMAGE_SIZE = 700
    FRAMESKIP_STEP = np.random(100) # 5
    # vs. "lost in details"
    STEPSKIP_DETECT = 10
    STEPSKIP_PLOT = STEPSKIP_DETECT # 10 >= FRAMESKIP_STEP == STEPSKIP_DETECT * k


# vs. single goal proficiency
class MetaObservation(base.MetaObservation):
    pass


# vs. too big search space
class PracticeSpace(base.PracticeSpace):
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True
    REWARD_ON_TERMINATE = 0 # vs. fear, losing confidence

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass


class PracticeTime(base.PracticeTime):    
    IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE = True # autom. decrease search-/practice-time
    GRACE_STEPS = 100
    REWARD_ON_TERMINATE = 0


class GoalRewardThreshold(base.GoalRewardThreshold):
    MAX_FRAC_DEFAULT = 0.05
    # vs. not finding goal (sparse rewards) (no direction/orientation: headless wandering)
    IS_NUDGING = True
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    # vs. too much repetitions for same start (few reps. for later difficults)
    IS_ENABLED = True


# vs. multi-dim. curse (exponentially many combinations)
# single/low-dim. comb-through (with basedims.)

# vs. reward neutralization? fear?
# no neg. rewards ("penalties")