
import numpy as np

from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from ..le_base.base_practice_env import BasePracticeEnv
from . import walker2d_dictobs_cfg as cfg


# ref https://www.youtube.com/watch?v=irkXnpZP89s
# records
# v1.0    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aab0ae/le-walker2d-v4/17-58-22/rl_model_finished
#   400k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43/rl_model_finished
#   500k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43_restored/rl_model_finished
#   400k, velo2, no practicemode    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/48779a8/le-walker2d-v4/00-14-47/rl_model_finished

# v2.0 - distance-dim., soft-/hard-terms
#   400k, soft-hard-terms   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/15-35-01/rl_model_finished
#   400k, soft-term-only    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/16-28-54/rl_model_finished
#   400k, hard-term-only    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/17-15-35/rl_model_finished

# v3.0 - refactored, very sparse
#   200k, reset threshold per ep.   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/19-50-45_restored_restored/rl_model_finished
#   300k, neg. reward       /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/08-07-14_restored_restored_restored_restored/rl_model_finished
#   400k, adaptive[0,1]     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/09-03-13_restored/rl_model_finished
#   400k, adaptive[0,0.5]   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/09-57-22_restored_restored/rl_model_finished
#   120k, noAdapt   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/11-06-36_restored/rl_model_finished

# v4.0 - strong weight diff.
#   120k, bidir. adaptive   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/19-31-37_restored_restored_restored/rl_model_finished

# v5 - shrinking adapt.
#   1.  200k, minmax[0,1]   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/13-16-01/rl_model_finished
#                        /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/14-12-21/rl_model_finished
#   2.  400k,               /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/14-12-21_restored/rl_model_finished
#   3.  400k, strongweight, halfAtHighestConv   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/15-38-32/rl_model_finished
#   4.  400k,               halfAtHighestConv,initShrink    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/18-16-05/rl_model_finished
#   5.  400k,               halfAtLowestDist    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/17-25-07/rl_model_finished
#   6.  400k, 0.2           noAdaptive, no trajhalv     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/592a508/le-walker2d-v4/21-02-44/rl_model_finished
#   7.  400k, 0.2           noAdaptive, halfAtLowestDist    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/592a508/le-walker2d-v4/20-06-07/rl_model_finished
#   8.  400k, minmax[0,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8d73cee/le-walker2d-v4/08-38-24/rl_model_finished              
#   9.  400k, minmax[0.05,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/638089b/le-walker2d-v4/09-46-08/rl_model_finished             
#   10.  400k, minmax[0.01,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/638089b/le-walker2d-v4/10-49-20/rl_model_finished             
# adaptive threshold (high max.) results in overall jerky, unsecure movements (unclear rewarding!), possibly better for universal training (rand. goals)? - no!
# adaptive threshold (high max.) also leads to large actor-/critic-losses
# need strongly different goal weighting!
# great improvement with trajHalv
# adapt. threshold: "perfecting" inside goal zone ("finetune", sufficient goal zone vs. perfect goal zone)

# v6 - random goals, no adapt
#   1.  400k, randomAllDimsNoWeights,Th0.2    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/21-57-09/rl_model_finished
#   2.  400k, randomWeightedDims,Th0.2        /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/23-35-09/rl_model_finished
#   3.  400k, randomWeightedDims,adaptiveTh   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/00-34-57/rl_model_finished
# random: slower, but generalizing (single dim, ie. velocity)

# v7(best)    0.2, noAdaptive, th-halfAtLowestDist
#   1M  /home/t500/tuhh/dsf/Scilab-RL/data/4779276/le-walker2d-v4/19-35-53/rl_model_finished
#       success!

# v8    minmax[0.05,0.2], th-halfAtLowestDist
#   1M  /home/t500/tuhh/dsf/Scilab-RL/data/8425c48/le-walker2d-v4/15-41-58/rl_model_finished
#       no goal-holding...

# v9 - incrementalist
        # incremental roadmap
#   1.  1M  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/9eb139a/le-walker2d-v4/18-04-17/rl_model_finished
#           reliable goal holding, faster initial learning (for more difficult goals)?
#   specialist
#       500k    home/t14/Documents/tuhh/dsf/Scilab-RL/data/9eb139a/le-walker2d-v4/20-31-35/rl_model_finished
#               fast targetted learning
#   incrementalist
#       500k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8ee9dc2/le-walker2d-v4/21-04-47/rl_model_finished
#               goal not reached
#   generalist
#       500k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a903811/le-walker2d-v4/15-12-18/rl_model_finished
#               slower learing

# v10   0.1, noAdapt, th-halfAtLowestDist, specialist, rewardNudge-personalRecord, velocityDim-only
#       100k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a2e4438/le-walker2d-v4/13-01-21/rl_model_finished
#               significant more efficient
#       noNudging
#           100k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a2e4438/le-walker2d-v4/13-16-52/rl_model_finished
#                   less efficient
#       noMetaObs
#           100k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a2e4438/le-walker2d-v4/14-42-20/rl_model_finished
#                   more efficient (less falling)
#       rewardNudge-personalRecord, metaObs
#           100k    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a2e4438/le-walker2d-v4/14-26-15/rl_model_finished
#                   best efficiency


# forming: goal + termination ("coaching")
# reward-trickling ("breadcrumbing")

# glossar
# GOALSPACE_DESIRED := goalstate -+ threshold
# (GOAL-)STATE_ACHIEVED := current state of step
# PRACTICE-/TRAINSPACE := all reasonable states to learn from

# TODO goal: time-dim. vs. infinite (non-episodic), stand-up? 

# https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
# https://gymnasium.farama.org/environments/mujoco/walker2d/
# src/custom_envs/maze/ant_env.py
# https://scilab-rl.github.io/Scilab-RL/wiki/Restore-a-saved-policy.html
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v4.py
# extend (vs wrapper)
class Walker2dDictObsEnv(BasePracticeEnv, Walker2dEnv):


    def __init__(self):
        Walker2dEnv.__init__(self, exclude_current_positions_from_observation=False)
        BasePracticeEnv.__init__(self, cfg)


    def get_achieved_goal(self, superobs):        
        distance, height, velocity, angle = superobs[0], superobs[1], superobs[9], superobs[2]
        # n_contact_after = self.data.ncon if self.ep_num_steps > 300 else 1
        angle_thigh = max(superobs[3], superobs[6])
        # is_moving_forward = velocity > 0.3 if self.ep_num_steps > 300 else 1

        return np.array((height, velocity, angle, angle_thigh))
        