

import mujoco
import numpy as np

# from gymnasium.envs.mujoco.pusher_v5 import PusherEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv
from gymnasium import spaces

ORDER_GOALMOMENTUM = 3

class MomFetchPushEnv(MujocoFetchPushEnv):


    def __init__(self, is_render=False, is_eval=False, submodels=None):
        self.is_render = is_render
        self.is_eval = is_eval
        self.goaldiffs = []
        self.goaldists = []
        self.rewardsum = 0
        self.outfile_goaldists = open('goaldists.dat', 'a')


        MujocoFetchPushEnv.__init__(self)


        observation_space = spaces.Box(-np.inf, np.inf, shape=(84,), dtype='float64')
        goal_space = spaces.Box(-np.inf, np.inf, shape=(ORDER_GOALMOMENTUM,), dtype='float64')



        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=goal_space,
                achieved_goal=goal_space,
            )
        )




    def _get_obs(self):
        observation = MujocoFetchPushEnv._get_obs(self)
        ob_box_achieved = observation['achieved_goal']
        ob_box_desired = observation['desired_goal']
        observation = observation['observation']
        if ob_box_desired.size == 0:
            ob_box_desired = np.zeros(3)

        goaldiff = np.array([])
        goaldiffs_recent = np.array([])
        goaldiffderivs = np.array([])

        goaldist = -1
        goaldists_recent = np.array([])
        goaldistderivs = np.array([])
        
        goalmomentum = np.array([])

        ob_gripper_pos = observation[0:3]

        obs_achieved = np.concatenate((ob_box_achieved, ob_gripper_pos))
        obs_desired = np.concatenate((ob_box_desired, ob_box_achieved))


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

        observation = np.append(observation, goaldiff)
        observation = np.append(observation, goaldiffs_recent)
        observation = np.append(observation, goaldiffderivs)

        observation = np.append(observation, goaldist)
        observation = np.append(observation, goaldists_recent)
        observation = np.append(observation, goaldistderivs)

        achieved_goal = np.array(goalmomentum)
        desired_goal = np.zeros(achieved_goal.shape)  # ignored

        dictobs = dict(
                observation=observation,
                achieved_goal=achieved_goal,
                desired_goal=desired_goal
            )

        return dictobs


    def step(self, action):
        (observation, reward, terminated, truncated, info) = MujocoFetchPushEnv.step(self, action)
        goalmomentum = observation['achieved_goal']

        reward = 0

        # soft vs. hard momentum
        if np.all(goalmomentum < 0):
            reward = 1
        # else:
        if np.all(goalmomentum > 0):
            reward = -1

        self.rewardsum += reward

        if terminated:
            reward = -1 # termination learning


        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(np.round(reward, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'rewardsum', str(np.round(self.rewardsum, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goaldist', str(np.round(self.goaldists[-1], 2)))
            human_viewer.render()

        return observation, reward, terminated, truncated, info


    def reset(self, seed, options):
        goaldists_mean = np.mean(self.goaldists)
        print(goaldists_mean)
        print(self.rewardsum)
        self.outfile_goaldists.write('%s\n' % (goaldists_mean))
        self.outfile_goaldists.flush()

        self.goaldiffs = []
        self.goaldists = []
        self.rewardsum = 0
        return MujocoFetchPushEnv.reset(self)
    


def normalize_unit_limit(val, min_val, max_val):
    # manual normalization (obs fairness)
    # "interval-shifting"
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    if max_val == min_val:
        return 0.5
    return (val - min_val) / (max_val - min_val)
