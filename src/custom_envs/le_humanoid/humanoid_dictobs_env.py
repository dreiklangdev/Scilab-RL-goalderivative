
import numpy as np

from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from ..le_base.base_practice_env import BasePracticeEnv
from . import humanoid_dictobs_cfg as cfg

# ref.
#   https://www.youtube.com/watch?v=gn4nRCC9TwQ&pp=ygUkcmVpbmZvcmNlbWVudCBsZWFybmluZyBodW1hbiB3YWxraW5n
#   https://www.youtube.com/watch?v=chMwFy6kXhs
#   https://www.youtube.com/watch?v=Qhb48SapCnU
#   https://www.ais.uni-bonn.de/nimbro/Humanoid/
#   https://www.ais.uni-bonn.de/nimbro/Humanoid/papers/TDP_NimbRo_AdultSize_2024.pdf
#   https://www.ais.uni-bonn.de/nimbro/Humanoid/papers/Specs_NimbRo_AdultSize_2024.pdf

#   1M, +traj.Halv +rand.Goals -adaptThresh    /mnt/t500/tuhh/dsf/Scilab-RL/data/600858d/le-humanoid-v4/09-17-30/rl_model_finished
#       holds goal reliably if forward walk (cant move backwards at all: needs different policy?) 
#   1M,            +rand.Goals(conformist)                         +goal-ext.obs. (goal-conscious)     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/31915dc/le-humanoid-v4/12-24-27/rl_model_finished
#       holds goal reliably, can walk backwards some time
#   1M,            +rand.Goals(generalist)     /mnt/t500/tuhh/dsf/Scilab-RL/data/e51f87f/le-humanoid-v4/20-17-14/rl_model_finished
#       wont learn equally, but reliably, even backwards (less mastery naturally at closer to edges/limits)


class HumanoidDictObsEnv(BasePracticeEnv, HumanoidEnv):


    def __init__(self):
        HumanoidEnv.__init__(self, exclude_current_positions_from_observation=True)
        # self.frame_skip = 10
        BasePracticeEnv.__init__(self, cfg)


    def extract_achieved_obs(self, superobs):
        height = superobs[0]
        x_velocity = superobs[22]
        return np.array((height, x_velocity))