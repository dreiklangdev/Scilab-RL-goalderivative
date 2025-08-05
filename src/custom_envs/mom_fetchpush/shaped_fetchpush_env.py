

import mujoco
import numpy as np

# from gymnasium.envs.mujoco.pusher_v5 import PusherEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv
from gymnasium import spaces


# TODO order vs. hardness vs. goaldiffObs vs. zscore

# noObs_dense_baseline (unshaped)
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/17-56-48/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/19-10-58/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-50/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-27/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-11-36/rl_model_finished

# noObs_dense_shaped (discounted)
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-12/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-17/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-20/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-24/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-27/rl_model_finished

# noObs_sparse_baseline (unshaped)
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-02/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-08/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-12/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-16/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-20/rl_model_finished

# noObs_sparse_shaped (discounted)
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-09/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-22/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-25/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-29/rl_model_finished
# /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-32/rl_model_finished

# sparse_shaped (discounted)



ORDER_GOALDYNAMICS = 3

class ShapedFetchPushEnv(MujocoFetchPushEnv):


    def __init__(self, is_render=False, is_eval=False, submodels=None):
        self.is_render = is_render
        self.is_eval = is_eval
        self.goaldeltas = []
        self.goaldists = []
        self.rewardsum = 0
        self.step_phi = np.zeros(ORDER_GOALDYNAMICS)
        self.step_goalderivs = np.zeros(ORDER_GOALDYNAMICS)
        self.outfile_goaldists = open('goaldists.dat', 'a')

        self.zs_scaler_goal = submodels['zs_scaler_goal']

        # sparse vs. dense
        MujocoFetchPushEnv.__init__(self, reward_type='sparse')


    def _get_obs(self):
        observation = MujocoFetchPushEnv._get_obs(self)
        ob_box_achieved = observation['achieved_goal']
        ob_box_desired = observation['desired_goal']
        obs = observation['observation']
        if ob_box_desired.size == 0:
            ob_box_desired = np.zeros(3)

        goaldelta = np.array([])
        goaldeltas_recent = np.array([])
        goaldeltas_velocity = np.array([])

        goaldist = -1
        goaldists_recent = np.array([])
        goaldistdeltas = np.array([])
        
        goalderivs = np.array([])

        ob_gripper_pos = obs[0:3]

        obs_achieved = np.concatenate((ob_box_achieved, ob_gripper_pos))
        obs_desired = np.concatenate((ob_box_desired, ob_box_achieved))

        if self.reward_type == "dense":
            # goal augmentation for "denser" env. (to match reward shaping goal and density)
            observation['achieved_goal'] = obs_achieved
            observation['desired_goal'] = obs_desired

        IS_NORMALIZE_Z_SCORE_GOAL = False
        if IS_NORMALIZE_Z_SCORE_GOAL:
            if not self.is_eval:
                self.zs_scaler_goal.partial_fit(obs_achieved.reshape(1, -1))
            obs_achieved = self.zs_scaler_goal.transform(obs_achieved.reshape(1, -1))[0]
            obs_desired = self.zs_scaler_goal.transform(obs_desired.reshape(1, -1))[0]


        goaldelta = obs_achieved - obs_desired
        self.goaldeltas.append(goaldelta)

        goaldist = np.linalg.norm(goaldelta, axis=-1)
        self.goaldists.append(goaldist)


        order = ORDER_GOALDYNAMICS
        if order > 0:
            goaldists_recent = np.array(self.goaldists[-(order + 1):]) # only enough recent goaldists for all orders
            goaldists_recent = np.pad(goaldists_recent, (max(0, order + 1 - len(goaldists_recent)),0)) # fill up with starting 0s if not enough
            for i in range(1, order + 1):
                goaldistdeltas = np.append(goaldistdeltas, np.diff(goaldists_recent, n=i, axis=0))
                goalderivs = np.append(goalderivs, goaldistdeltas[-1]) # front

        # obs vs. no-obs
        obs = np.append(obs, goaldist)
        obs = np.append(obs, goalderivs)
        observation['observation'] = obs

        self.step_goalderivs = goalderivs
        self.step_phi = -np.linalg.norm(np.concatenate(([goaldist], goalderivs[:-1])))

        return observation


    def step(self, action):
        phi_prev = self.step_phi
        (observation, reward, terminated, truncated, info) = MujocoFetchPushEnv.step(self, action)
        phi = self.step_phi

        # potential-based shaping (undiscounted)
        reward -= self.step_goalderivs

        # vs. potential-based shaping (discounted)
        # reward += 0.99 * phi - phi_prev

        # if info['is_success']:
            # reward = 1 # success learning

        self.rewardsum += reward

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

        self.goaldeltas = []
        self.goaldists = []
        self.rewardsum = 0
        self.step_phi = np.zeros(ORDER_GOALDYNAMICS)
        self.step_goalderivs = np.zeros(ORDER_GOALDYNAMICS)
        return MujocoFetchPushEnv.reset(self)
    


def normalize_unit_limit(val, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return (val - min_val) / (max_val - min_val)
