
import numpy as np

from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from ..le_base import base_practice_env
from . import humanoid_dictobs_cfg as cfg

#   1M      /mnt/t500/tuhh/dsf/Scilab-RL/data/600858d/le-humanoid-v4/09-17-30/rl_model_finished
#       +traj.Halv +rand.Goals(accidently) (-adaptThreshold)
#   holds goal reliably if forward walk (backward walk differs to much: needs different policy?) 

class HumanoidDictObsEnv(base_practice_env.BasePracticeEnv, HumanoidEnv):


    def __init__(self):
        HumanoidEnv.__init__(self, exclude_current_positions_from_observation=True)
        base_practice_env.BasePracticeEnv.__init__(self, cfg)


    def _get_obs(self):
        observation = HumanoidEnv._get_obs(self)
        height = observation[0]
        x_velocity = observation[22]

        achieved_goal = np.array((height, x_velocity))
        achieved_goal_norm = self._normalize(achieved_goal, cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[1])
        desired_goal_norm = self._normalize(self.desired_goal, cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[1])

        dictobs = dict(
                observation=observation,
                achieved_goal=achieved_goal_norm,
                desired_goal=desired_goal_norm,
            )
        
        return dictobs

    