
from ..le_base import base_practice_cfg as base
import numpy as np


class General(base.General):
    MAX_STEPS_EPISODE_TRUNCATION = 1500

    WORLD_OBS_DIFFS_ORDER = 6

    LANDMARK_GROUPS = [ # redundant duplicates?
        # [0],                            # nose
        # [8, 6, 5, 4, 1, 2, 3, 7],       # face
        # [10, 9],                       # mouth    
        [11, 13, 15, 17, 19, 15, 21],  # right arm
        [12, 14, 16, 18, 20, 16, 22],  # left arm
        # [11, 23, 25, 27, 29, 31, 27],  # right body side
        # [12, 24, 26, 28, 30, 32, 28],  # left body side
        # [11, 12],                      # shoulder
        # [29, 30],                      # feet
        # [23, 24],                      # waist
    ]
    LANDMARK_GROUPS_FLAT = np.hstack(LANDMARK_GROUPS) if LANDMARK_GROUPS else []


    MJBODY_TO_MPPOSE = [None, None, None, None, None, None, None, None, None, None, None, 'l_sho_pitch_link', 'r_sho_pitch_link', 'l_sho_roll_link', 'r_sho_roll_link', 'l_el_link', 'r_el_link', None, None, None, None, None, None, 'l_hip_pitch_link', 'r_hip_pitch_link', 'l_knee_link', 'r_knee_link', 'l_ank_pitch_link', 'r_ank_pitch_link']
    mjlandmarks = np.where(np.array(MJBODY_TO_MPPOSE) != None)[0]
    IDS_LANDMARKS_FILTERED = np.intersect1d(mjlandmarks, LANDMARK_GROUPS_FLAT)

    groups_filtered = LANDMARK_GROUPS.copy()
    for i in range(len(groups_filtered)):
        groups_filtered[i] = np.array(groups_filtered[i])
        groups_filtered[i] = groups_filtered[i][np.isin(groups_filtered[i], IDS_LANDMARKS_FILTERED)]
    LANDMARK_GROUPS_FILTERED = groups_filtered


    NUM_OBSERVATION_DIMS_VISUAL_DETECTION = IDS_LANDMARKS_FILTERED.size * 3 if LANDMARK_GROUPS else 0
    RENDER_IMAGE_SIZE = 1000
    FRAMESKIP_STEP = 3 # should resemble reality speed for detection
    # vs. "lost in details"
    STEPSKIP_DETECT = 9999999999 # 1 # 10 # may need to delay fep_goaldist_init (first detected pose may be unstable/in-the-air)
    STEPSKIP_PLOT = STEPSKIP_DETECT # >= FRAMESKIP_STEP == STEPSKIP_DETECT * k

    MAX_LIVES = 3


# vs. single goal proficiency
class MetaObservation(base.MetaObservation):
    IS_ENABLED = True


class DbObservation(base.DbObservation):
    IS_ACTIONDB_ENABLED = False
    ACTIONDB_SIMILARITY_THRESHOLD = 0.1 # below


# termination shaping (vs. too big search space)
class PracticeSpace(base.PracticeSpace):
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True
    STEPS_INVINCIBLE_SPAWN = 0 # eg. if instable start (falling)
    REWARD_ON_TERMINATE = 0 # vs. fear, losing confidence

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass


class PracticeTime(base.PracticeTime):    
    IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE = True # autom. decrease search-/practice-time
    REWARD_ON_TERMINATE = 0


class GoalRewardThreshold(base.GoalRewardThreshold):
    # MAX_FRAC_DEFAULT = 0.05 # should be greater than drift/noise
    MAX_FRAC_DEFAULT = 0.15
    # vs. not finding goal (sparse rewards) (no direction/orientation: headless wandering)
    IS_NUDGING = True
    IS_ADAPTIVE = False

    IS_SPARSE_MODE_TOGGLE_ENABLED = True
    MIN_STEPS_FOR_SPARSE_MODE_TOGGLE = 10 # 10 # makes it rather worse if enabled?


class TrajectoryHalving(base.TrajectoryHalving):
    # vs. too much repetitions for same start (few reps. for later difficults)
    # more relevant for envs where goal-state is far from the start
    IS_ENABLED = False
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE


# for better reward signal (single-/low-dim., piece-wise emphasize)
# vs. multi-dim. curse (exponentially many combinations)
# ndim. goal single/low-dim. comb-through (with basedims.?)

# vs. reward neutralization? fear?
# no neg. rewards ("penalties")