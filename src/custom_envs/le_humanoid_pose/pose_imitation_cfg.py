
from ..le_base import base_practice_cfg as base
import numpy as np


class General(base.General):
    LANDMARK_GROUPS = [ # redundant duplicates?
        # [0],                            # nose
        # [8, 6, 5, 4, 1, 2, 3, 7],       # face
        # [10, 9],                       # mouth    
        # [11, 13, 15, 17, 19, 15, 21],  # right arm
        # [12, 14, 16, 18, 20, 16, 22],  # left arm
        # [11, 23, 25, 27, 29, 31, 27],  # right body side
        # [12, 24, 26, 28, 30, 32, 28],  # left body side
        # [11, 12],                      # shoulder
        # [29, 30],                      # feet
        # [23, 24],                      # waist
    ]
    LANDMARK_GROUPS_FLAT = np.hstack(LANDMARK_GROUPS) if LANDMARK_GROUPS else []

    OBSERVATION_DIMS_VISUAL_DETECTION = LANDMARK_GROUPS_FLAT.size * 3 if LANDMARK_GROUPS else 0
    RENDER_IMAGE_SIZE = 1000
    FRAMESKIP_STEP = 3 # should resemble reality speed for detection
    # vs. "lost in details"
    STEPSKIP_DETECT = 999999999 # 1 # 10 # may need to delay fep_goaldist_init (first detected pose may be unstable/in-the-air)
    STEPSKIP_PLOT = STEPSKIP_DETECT # >= FRAMESKIP_STEP == STEPSKIP_DETECT * k

    MAX_LIVES = 0


# vs. single goal proficiency
class MetaObservation(base.MetaObservation):
    IS_ENABLED = True


# termination shaping (vs. too big search space)
class PracticeSpace(base.PracticeSpace):
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True
    STEPS_INVINCIBLE_SPAWN = 25 # eg. if instable start (falling)
    REWARD_ON_TERMINATE = 0 # vs. fear, losing confidence

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass


class PracticeTime(base.PracticeTime):    
    IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE = True # autom. decrease search-/practice-time
    REWARD_ON_TERMINATE = 0


class GoalRewardThreshold(base.GoalRewardThreshold):
    # MAX_FRAC_DEFAULT = 0.05 # should be greater than drift/noise
    MAX_FRAC_DEFAULT = 0.05
    # vs. not finding goal (sparse rewards) (no direction/orientation: headless wandering)
    IS_NUDGING = True
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    # vs. too much repetitions for same start (few reps. for later difficults)
    # good for envs where goal-state is far from the beginning
    # bad for envs where start-state is in/near goalzone (too early savepoints)
    IS_ENABLED = True
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE


# vs. multi-dim. curse (exponentially many combinations)
# single/low-dim. comb-through (with basedims.)

# vs. reward neutralization? fear?
# no neg. rewards ("penalties")