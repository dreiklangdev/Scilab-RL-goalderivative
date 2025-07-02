
from ..le_base import base_practice_cfg as base
import numpy as np


class General(base.General):
    # for sample eff.: avoid "every step feel the same" (homogenity) (wasted steps: "zero-steps" (no reward) vs. "indifferent-steps" (too fine rewards, little distinction between))

    # derivs ~ "curvature" info

    # restricting/specializing (slower, "safer")
    WORLD_DERIV_ORDERS = 2 # "smoothness-factor"
    # exploring/generalizing (slower, "freer")
    GOAL_DERIV_ORDERS = 3 # "sparsity-factor"

    OBS_REWARD_HISTORY_LENGTH = 0
    REWARD_DERIV_ORDERS = 0 # only for dense rewards

    LANDMARK_GROUPS = [
        [0, 1, 2, 3, 4], # thumb
        [5, 6, 7, 8], # index
        [9, 10, 11, 12], # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20], # pinky
    ]
    LANDMARK_GROUPS = [ # wrist and fingerpoints only # geom
        [0, 5, 17], # palm (for orientation)
        # [0, 5, 9, 13, 17], # palm (for orientation)
        [0, 4], # thumb
        [0, 8], # index
        [0, 12], # middle
        [0, 16], # ring
        [0, 20], # pinky
    ]
    # LANDMARK_GROUPS = [ # wrist and fingerpoints only
    #     [0, 7], # index
    # ]

    LANDMARK_GROUPS_FLAT = np.hstack(LANDMARK_GROUPS) if LANDMARK_GROUPS else []

    # MJBODY_TO_MPPOSE = ['rh_wrist',
    #                     'rh_thproximal', 'rh_thmiddle', 'rh_thdistal', None,
    #                     'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal', None,
    #                     'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal', None,
    #                     'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal', None,
    #                     'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal', None,
    #                     ]
    # MJBODY_TO_MPPOSE = ['wrist', # body
    #                     'thproximal', 'thmiddle', 'thdistal', None,
    #                     'ffproximal', 'ffmiddle', 'ffdistal', None,
    #                     'mfproximal', 'mfmiddle', 'mfdistal', None,
    #                     'rfproximal', 'rfmiddle', 'rfdistal', None,
    #                     'lfproximal', 'lfmiddle', 'lfdistal', None,
    #                     ]
    MJBODY_TO_MPPOSE = ['V_wrist', # geom
                        'V_thbase', 'V_thproximal', 'V_thmiddle', 'V_thdistal',
                        'V_ffknuckle', 'V_ffproximal', 'V_ffmiddle', 'V_ffdistal', 
                        'V_mfknuckle', 'V_mfproximal', 'V_mfmiddle', 'V_mfdistal',
                        'V_rfknuckle', 'V_rfproximal', 'V_rfmiddle', 'V_rfdistal',
                        'V_lfknuckle', 'V_lfproximal', 'V_lfmiddle', 'V_lfdistal',
                        ]
    mjlandmarks = np.where(np.array(MJBODY_TO_MPPOSE) != None)[0]
    IDS_LANDMARKS_FILTERED = np.intersect1d(mjlandmarks, LANDMARK_GROUPS_FLAT)

    groups_filtered = LANDMARK_GROUPS.copy()
    for i in range(len(groups_filtered)):
        groups_filtered[i] = np.array(groups_filtered[i])
        groups_filtered[i] = groups_filtered[i][np.isin(groups_filtered[i], IDS_LANDMARKS_FILTERED)]
    LANDMARK_GROUPS_FILTERED = groups_filtered


    NUM_OBSERVATION_DIMS_VISUAL_DETECTION = IDS_LANDMARKS_FILTERED.size * 3 if LANDMARK_GROUPS else 0
    
    RENDER_IMAGE_SIZE = 1000
    FRAMESKIP_STEP = 3 # action frequency
    # vs. "lost in details"
    STEPSKIP_DETECT = 50 # 50 # 1 # 10 # may need to delay fep_goaldist_init (first detected pose may be unstable/in-the-air)


# vs. single goal proficiency
class MetaObservation(base.MetaObservation):
    IS_ENABLED = False


# termination shaping (vs. too big search space)
class PracticeSpace(base.PracticeSpace):
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True
    STEPS_INVINCIBLE_SPAWN = 0 # eg. if instable start (falling)
    REWARD_ON_TERMINATE = 0 # vs. fear, losing confidence

    MAX_STEPS_EPISODE_TRUNCATION = 10000
    MAX_DIVERGENT_STEPS = 500 # 500

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass


class GoalRewardThreshold(base.GoalRewardThreshold):
    MAX_FRAC_DEFAULT = 0.3
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    # vs. too much repetitions for same start (few reps. for later difficults)
    # more relevant for envs where goal-state is far from the start
    IS_ENABLED = False
    STRAT = base.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
    MAX_LIVES = 5 # 3


# for better reward signal (single-/low-dim., piece-wise emphasize)
# vs. multi-dim. curse (exponentially many combinations)
# ndim. goal single/low-dim. comb-through (with basedims.?)

# vs. reward neutralization? fear?
# no neg. rewards ("penalties")