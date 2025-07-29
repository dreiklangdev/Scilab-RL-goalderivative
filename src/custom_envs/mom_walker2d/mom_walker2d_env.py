

import mujoco
import numpy as np

from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium import spaces


# 150k  vanilla     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/695d11a/mom-walker2d-v5/15-00-32/rl_model_finished
# 150k  goalHeightVelo, goalMomRewardsOnly, goaldiffObsAdded

ORDER_GOALMOMENTUM = 4

class MomWalker2dEnv(Walker2dEnv):


    def __init__(self, is_render=False, is_eval=False, submodels=None):
        Walker2dEnv.__init__(self)
        self.is_render = is_render
        self.is_eval = is_eval
        self.goaldiffs = []
        self.goaldists = []
        self.rewardsum = 0
        self.x_velocity = 0
        self.outfile_goaldists = open('goaldists.dat', 'a')

        obspace_total_dims = self.observation_space.shape[0] # super
        observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype='float64')
        goal_space = spaces.Box(-np.inf, np.inf, shape=(ORDER_GOALMOMENTUM,), dtype='float64')

        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=goal_space,
                achieved_goal=goal_space,
            )
        )


    def _get_obs(self):
        observation = Walker2dEnv._get_obs(self)

        goaldiff = np.array([])
        goaldiffs_recent = np.array([])
        goaldiffderivs = np.array([])

        goaldist = -1
        goaldists_recent = np.array([])
        goaldistderivs = np.array([])
        
        goalmomentum = np.array([])



        # obs_achieved = np.array([observation[0], observation[8], observation[9]]) # height, velocity_x, velocity_height
        # obs_desired = np.array([1.1, 1.5, 0])

        # obs_achieved = np.array([observation[0], observation[1], observation[10]]) # velocity
        # obs_achieved = np.array([self.x_velocity]) # velocity
        # obs_achieved = np.array([observation[0], observation[1], observation[9]]) # velocity

        x, z, angle = self.data.qpos[0:3]

        ob_x = normalize_unit_limit(x, 0, 10)

        obs_achieved = np.array([ob_x, z]) # velocity
        obs_desired = np.array([1, 1.1])


        goaldiff = obs_achieved - obs_desired
        self.goaldiffs.append(goaldiff)

        goaldist = np.linalg.norm(goaldiff, axis=-1)
        self.goaldists.append(goaldist)


        goalderiv_orders = ORDER_GOALMOMENTUM
        if goalderiv_orders > 0:
            goaldiffs_recent = np.array(self.goaldiffs[-(goalderiv_orders + 1):]) # only enough recent goaldists for all orders
            goaldiffs_recent = np.pad(goaldiffs_recent, ((max(0, goalderiv_orders + 1 - len(goaldiffs_recent)),0), (0,0))) # fill up with starting 0s if not enough
            goaldiffderivs = np.diff(goaldiffs_recent, axis=0)

            goaldists_recent = np.array(self.goaldists[-(goalderiv_orders + 1):]) # only enough recent goaldists for all orders
            goaldists_recent = np.pad(goaldists_recent, (max(0, goalderiv_orders + 1 - len(goaldists_recent)),0)) # fill up with starting 0s if not enough
            for i in range(1, goalderiv_orders + 1):
                goaldistderivs = np.append(goaldistderivs, np.diff(goaldists_recent, n=i, axis=0))
                goalmomentum = np.append(goalmomentum, goaldistderivs[-1]) # front


        # obs. combination vs. diff. obs. only? (requires mom. based rewards?)
        # observation = np.array([])

        # observation = np.append(observation, goaldiff)
        # observation = np.append(observation, goaldiffs_recent)
        # observation = np.append(observation, goaldiffderivs)

        # observation = np.append(observation, goaldist)
        # observation = np.append(observation, goaldists_recent)
        # observation = np.append(observation, goaldistderivs)

        achieved_goal = np.array(goalmomentum)
        desired_goal = np.zeros(achieved_goal.shape)  # ignored

        dictobs = dict(
                observation=observation,
                achieved_goal=achieved_goal,
                desired_goal=desired_goal
            )

        return dictobs


    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        self.x_velocity = x_velocity

        observation = self._get_obs()
        # reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        truncated = False
        info = {}
        # info = {
        #     "x_position": x_position_after,
        #     "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        #     "x_velocity": x_velocity,
        #     **reward_info,
        # }

        goalmomentum = observation['achieved_goal']
        
        reward = 0

        if np.any(goalmomentum < 0):
            reward = 1
        elif np.any(goalmomentum > 0):
            reward = -1
            print('PENALTY')
        elif np.mean(np.sign(goalmomentum)) < 0:
            reward = 1
        elif np.mean(np.sign(goalmomentum)) > 0:
            reward = -1
            print('PENALTY')

        self.rewardsum += reward

        if terminated:
            print('TERMINATED\n')
            reward = -1
            # if self.rewardsum > 0:
            #     reward = -self.rewardsum 


        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(np.round(reward, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'rewardsum', str(np.round(self.rewardsum, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goaldist', str(np.round(self.goaldists[-1], 2)))
            human_viewer.render()

        return observation, reward, terminated, truncated, info
    

    def reset_model(self):
        goaldists_mean = np.mean(self.goaldists)
        print(goaldists_mean)
        print(self.rewardsum)
        self.outfile_goaldists.write('%s\n' % (goaldists_mean))
        self.outfile_goaldists.flush()

        self.goaldiffs = []
        self.goaldists = []
        self.rewardsum = 0
        self.x_velocity = 0
        return super().reset_model()
    


def normalize_unit_limit(val, min_val, max_val):
    # manual normalization (obs fairness)
    # "interval-shifting"
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    if max_val == min_val:
        return 0.5
    return (val - min_val) / (max_val - min_val)
