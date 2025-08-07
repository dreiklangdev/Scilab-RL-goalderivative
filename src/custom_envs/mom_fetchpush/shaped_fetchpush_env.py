

import mujoco
import numpy as np

# from gymnasium.envs.mujoco.pusher_v5 import PusherEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv


# TODO order vs. hardness vs. goaldiffObs vs. zscore

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
        obs_box_desired = observation['desired_goal']
        obs = observation['observation']
        if obs_box_desired.size == 0:
            obs_box_desired = np.zeros(3)

        goaldelta = np.array([])
        goaldeltas_recent = np.array([])
        goaldeltas_velocity = np.array([])

        goaldist = -1
        goaldists_recent = np.array([])
        goaldistdeltas = np.array([])
        
        goalderivs = np.array([])

        ob_gripper_pos = obs[0:3]

        # denser shaping
        obs_achieved = np.concatenate((ob_box_achieved, ob_gripper_pos))
        if obs_box_desired.size == 3:
            obs_desired = np.concatenate((obs_box_desired, ob_box_achieved))
        else:
            # already modified
            obs_desired = obs_box_desired

        # obs_achieved = ob_box_achieved
        # obs_desired = ob_box_desired

        # goal modification for "denser" env. (to match reward shaping goal and density)
        IS_GOAL_MOD = False
        if IS_GOAL_MOD and self.reward_type == 'dense':
            observation['achieved_goal'] = obs_achieved
            observation['desired_goal'] = obs_desired
            self.goal = obs_desired


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

        # self.step_phi = -np.linalg.norm(np.concatenate(([goaldist], goalderivs[:-1])))
        # self.step_phi = -np.linalg.norm(goalderivs)
        # self.step_phi = np.all(goalderivs < 0)

        # boolphi
        self.step_phi = 0
        if np.all(goalderivs < 0):
            self.step_phi = 1
        elif np.all(goalderivs > 0):
            self.step_phi = -1

        return observation


    def step(self, action):
        phi_prev = self.step_phi
        (observation, reward, terminated, truncated, info) = MujocoFetchPushEnv.step(self, action)
        phi = self.step_phi

        IS_POS_SPARSE_ENV = True
        if IS_POS_SPARSE_ENV and self.reward_type == 'sparse':
            reward += 1 # (0,1) instead of (-1,0) to improve shaping influence? (else inhibition: explore all left(-1) vs. exploit already found(1))

        # potential-based norm-shaping (undiscounted)
        # reward -= self.step_goalderivs

        # vs. potential-based norm-shaping (discounted)
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
        # print(goaldists_mean)
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
