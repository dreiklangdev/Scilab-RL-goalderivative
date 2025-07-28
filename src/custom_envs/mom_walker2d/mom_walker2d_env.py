


import numpy as np

from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium import spaces


class MomWalker2dEnv(Walker2dEnv):


    def __init__(self, is_render=False, is_eval=False, submodels=None):
        Walker2dEnv.__init__(self)
        self.is_render = is_render
        self.is_eval = is_eval

        obspace_total_dims = self.observation_space.shape[0] # super
        observation_space = spaces.Box(-np.inf, np.inf, shape=(obspace_total_dims,), dtype='float64')
        goal_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float64')

        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=goal_space,
                achieved_goal=goal_space,
            )
        )


    def _get_obs(self):
        observation = Walker2dEnv._get_obs(self)

        dictobs = dict(
                observation=observation,
                achieved_goal=0,
                desired_goal=0,
            )

        return dictobs


    def step(self, action):
        (observation, reward, terminated, truncated, info) = Walker2dEnv.step(self, action)
        
        if self.is_render:
            self.render_mode = 'human'
            self.render()

        return observation, reward, terminated, truncated, info