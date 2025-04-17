
import numpy as np

from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from ..le_base.base_practice_env import BasePracticeEnv
from . import humanoid_dictobs_cfg as cfg

#   1M, +traj.Halv +rand.Goals -adaptThresh    /mnt/t500/tuhh/dsf/Scilab-RL/data/600858d/le-humanoid-v4/09-17-30/rl_model_finished
#       holds goal reliably if forward walk (cant move backwards at all: needs different policy?) 
#   1M,            +rand.Goals(conformist)                         +goal-ext.obs. (goal-conscious)     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/31915dc/le-humanoid-v4/12-24-27/rl_model_finished
#       holds goal reliably, can walk backwards some time
#   1M,            +rand.Goals(generalist)     /mnt/t500/tuhh/dsf/Scilab-RL/data/e51f87f/le-humanoid-v4/20-17-14/rl_model_finished
#       wont learn equally, but reliably, even backwards (less mastery at closer to edges/limits) (naturally)

class HumanoidDictObsEnv(BasePracticeEnv, HumanoidEnv):


    def __init__(self):
        HumanoidEnv.__init__(self, exclude_current_positions_from_observation=True)
        BasePracticeEnv.__init__(self, cfg)


    def _get_obs(self):
        observation = BasePracticeEnv._get_obs(self)
        height = observation[0]
        x_velocity = observation[22]

        achieved_goal = np.array((height, x_velocity))
        achieved_goal_norm = self._normalize(achieved_goal, cfg.PracticeSpace.d[0], cfg.PracticeSpace.d[1])
        desired_goal_norm = self._normalize(self.desired_goal, cfg.PracticeSpace.d[0], cfg.PracticeSpace.d[1])

        dictobs = dict(
                observation=observation,
                achieved_goal=achieved_goal_norm,
                desired_goal=desired_goal_norm,
            )
        
        return dictobs

    