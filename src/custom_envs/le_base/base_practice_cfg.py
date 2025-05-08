
from enum import Enum
import numpy as np

# remote
# ssh t500@192.168.178.36
# sudo sshfs -o allow_other,default_permissions t500@192.168.178.36:/home/t500 /mnt/t500
# tmux
# MUJOCO_GL=egl %python ...%

# TODO transform/move to dict? (immutable as configs should be)

class General:
    EPISODE_TRUNCATION_STEPS_MAX = 1000
    EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN = 0.95


class MetaObservation:
    IS_ENABLED = True


# as general basic training? (base policy)
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
    # only works for similar goals (same dim., same sign)
    # jack-of-all-trades (but non perfectly) vs. perfectionist
    IS_TERMINATION_IF_OUTSIDE = True # radically decrease state-/searchspace
    REWARD_IF_OUTSIDE = 0
    
    d = None
    labels = None

    def init_dims(dims, labels):
        PracticeSpace.d = np.array(dims)
        PracticeSpace.labels = np.array(labels)

        diameter = np.linalg.norm(PracticeSpace.d[1] - PracticeSpace.d[0])
        diameter_normed = np.sqrt(PracticeSpace.d.shape[1])
        mode = np.linalg.norm(PracticeSpace.d[2])
        mode_ratio = mode / diameter
        radius_ratio = max(mode_ratio, 1 - mode_ratio)
        PracticeSpace.radius = radius_ratio * diameter
        PracticeSpace.radius_normed = radius_ratio * diameter_normed

    # closer to edge => less (surrounding) exposure => less mastery (ie. wont learn practice limits at all)
    # mode: always at midpoint? 
    # example
    # base.PracticeSpace.init_dims([
    #     [1.0,        -2.0],  # min
    #     [2.0,        3.0],   # max
    #     [1.3,        -1.0],  # mode
    #     [0.05,       1.0]    # weight [0,1]
    # ], ['height',   'velocity'])


    class RandomGoalSampling:
        class Strat(Enum):
            GENERALIST = 0
            CONFORMIST = 1
            SPECIALIST = 2 # mode only (non-random)
        STRAT = Strat.GENERALIST


class GoalRewardThreshold:
    # "breadcrumbing"
    # rewards: (sparse > freq.)
    #   too frequent => no movement (idleness, fast-narrow conv.)
    #   too sparse => no improvement (randomness, slow-broad conv.)
    #   too painful => no courage (fearful, no conv.)

    # rewarding: static, predictable > adaptive, dynamic?
    IS_ADAPTIVE = False
    ADAPTION_PADDING = 0.3

    IS_NUDGING = True


    # smaller: faster reach
    # too small: will never hold?
    MIN_FRAC = 0.05 # REWARD TOLERANCE ( > 0: better/easier for goal-holding (at all? "nobody is perfect"))
    MAX_FRAC_DEFAULT = 0.1 # decaying?
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


class CfgAssertion:
    assert 0 <= GoalRewardThreshold.MAX_FRAC_DEFAULT <= 1, f'normalized threshold must be in [0,1], is: {GoalRewardThreshold.MAX_FRAC_DEFAULT}'
