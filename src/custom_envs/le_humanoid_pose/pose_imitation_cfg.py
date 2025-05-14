
from ..le_base import base_practice_cfg as base


class General(base.General):
    OBSERVATION_DIMS_VISUAL_DETECTION = 99
    RENDER_IMAGE_SIZE = 400
    FRAMESKIP_STEP = 5
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