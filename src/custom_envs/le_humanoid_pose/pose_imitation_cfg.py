
from ..le_base import base_practice_cfg as base


class General(base.General):
    OBSERVATION_DIMS_VISUAL_DETECTION = 99
    RENDER_IMAGE_SIZE = 480
    FRAMESKIP_STEP = 5
    STEPSKIP_DETECT = 2
    STEPSKIP_PLOT = 2 # 10 >= FRAMESKIP_STEP_DETECT


class MetaObservation(base.MetaObservation):
    pass


class PracticeSpace(base.PracticeSpace):
    IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE = True
    REWARD_ON_TERMINATE = -1

    class RandomGoalSampling(base.PracticeSpace.RandomGoalSampling):
        pass


class PracticeTime(base.PracticeTime):    
    IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE = True # autom. decrease search-/practice-time
    GRACE_STEPS = 100
    REWARD_ON_TERMINATE = -1


class GoalRewardThreshold(base.GoalRewardThreshold):
    MAX_FRAC_DEFAULT = 0.1 # TODO use
    IS_NUDGING = True
    IS_ADAPTIVE = False


class TrajectoryHalving(base.TrajectoryHalving):
    IS_ENABLED = False
