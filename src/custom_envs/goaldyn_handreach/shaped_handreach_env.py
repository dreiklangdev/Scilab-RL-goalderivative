

import mujoco
import numpy as np

from gymnasium_robotics.envs.shadow_dexterous_hand.reach import MujocoHandReachEnv


# TODO order vs. hardness vs. goaldiffObs vs. zscore


ORDER_GOALDYNAMICS = 3

class ShapedHandReachEnv(MujocoHandReachEnv):


    def __init__(self, is_render=False, is_eval=False, submodels=None):
        self.is_render = is_render
        self.is_eval = is_eval
        self.subdists = []
        self.goaldists = []
        self.goalprogress = 0
        self.rewardsum = 0
        self.step_phi = np.zeros(ORDER_GOALDYNAMICS)
        self.step_goalderivs = np.zeros(ORDER_GOALDYNAMICS)
        self.outfile_goalprogresses = open('goalprogresses.dat', 'a')

        MujocoHandReachEnv.__init__(self, reward_type='sparse')


    def _get_obs(self):
        observation = MujocoHandReachEnv._get_obs(self)
        obs_achieved = observation['achieved_goal']
        obs_desired = observation['desired_goal']
        obs = observation['observation']
        if obs_desired.size == 0:
            obs_desired = np.zeros(15)

        subdists = np.array([])
        subdists_recent = np.array([])
        subdists_velos_recent = np.array([])
        subdists_accs_recent = np.array([])

        goaldist = -1
        goaldists_recent = np.array([])
        goaldistdiffs = np.array([])

        goalderivs = np.array([])

        subdists = obs_achieved - obs_desired
        self.subdists.append(subdists)

        goaldist = np.linalg.norm(subdists, axis=-1)
        self.goaldists.append(goaldist)


        order = ORDER_GOALDYNAMICS
        if order > 0:
            subdists_recent = np.array(self.subdists[-(order + 1):]) # only enough recent goaldists for all orders
            subdists_recent = np.pad(subdists_recent, ((max(0, order + 1 - len(subdists_recent)),0), (0,0))) # fill up with starting 0s if not enough
            subdists_velos_recent = np.diff(subdists_recent, axis=0)
            subdists_accs_recent = np.diff(subdists_velos_recent, axis=0)
            subdists_jerks_recent = np.diff(subdists_accs_recent, axis=0)

            goaldists_recent = np.array(self.goaldists[-(order + 1):]) # only enough recent goaldists for all orders
            goaldists_recent = np.pad(goaldists_recent, (max(0, order + 1 - len(goaldists_recent)),0)) # fill up with starting 0s if not enough
            for i in range(1, order + 1):
                goaldistdiffs = np.append(goaldistdiffs, np.diff(goaldists_recent, n=i, axis=0))
                goalderivs = np.append(goalderivs, goaldistdiffs[-1]) # front


        # obs reduce
        IS_OBS_REDUCE = True
        if IS_OBS_REDUCE:
            obs = np.array([])

        # full-obs
        IS_OBS_AUG = True
        if IS_OBS_AUG:

            # k = 10 # override past history length? (might be best at order+1 anyway)
            # goaldists_recent = np.pad(self.goaldists[-k:], (max(0, k - len(self.goaldists[-k:])),0))
            # subdists_recent = np.pad(self.subdists[-k:], ((max(0, k - len(self.subdists[-k:])),0), (0,0))) # fill up with starting 0s if not enough

            # maingoal space
            obs = np.append(obs, goaldist) # +c constant (offset) # at least one recent (NN can infer more recent goaldists. by integration with given goalderivs.)
            # obs = np.append(obs, goaldists_recent) # recent positions for inferring goalderivs. on its own (but might affect generality,speed)
            obs = np.append(obs, goalderivs) # given pre-computed goaldynamics might improve generality,speed

            # subgoal space
            obs = np.append(obs, subdists)
            # obs = np.append(obs, subdists_recent)
            # TODO more subderivs? (improves stability,convergence,speed?! how about generality?)
            # obs = np.append(obs, subdists_velos_recent)
            # obs = np.append(obs, subdists_accs_recent)
            # obs = np.append(obs, subdists_jerks_recent)
            obs = np.append(obs, subdists_velos_recent[-1])
            obs = np.append(obs, subdists_accs_recent[-1])
            obs = np.append(obs, subdists_jerks_recent[-1])


        # dgs-obs
        IS_OBS_DGS = False
        if IS_OBS_DGS:
            obs = np.append(obs, goaldist)
            obs = np.append(obs, goalderivs)

        observation['observation'] = obs

        self.step_goalderivs = goalderivs

        # boolphi (only shaping!)
        self.step_phi = 0
        if np.all(goalderivs[::2] < 0) and np.all(goalderivs[1::2] > 0): # reward slow down every derivative towards goal (instead of wrong old version: slowing down only every second derivative)
            self.step_phi = 1
        elif np.all(goalderivs > 0):
            self.step_phi = -1

        return observation
    


    # is also used by HER (multi-dim. args.)
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        if achieved_goal.ndim > 1:
            # recursive for replay buffer
            return np.array([self.compute_reward(ag, dg, i) for (ag, dg, i) in zip(achieved_goal, desired_goal, info)])

        reward = MujocoHandReachEnv.compute_reward(self,achieved_goal, desired_goal, info)

        # (0,1) instead of (-1,0) to improve shaping influence? (else inhibition: explore all left(-1) vs. exploit already found(1))
        IS_POS_SPARSE_ENV = False
        if IS_POS_SPARSE_ENV and self.reward_type == 'sparse':
            reward += 1

        # potential-based shaping (discounted)
        IS_REWARD_SHAPING = False
        if IS_REWARD_SHAPING:
            if 'phi_prev' in info.keys():
                phi_prev = info['phi_prev']
                phi = info['phi']
                reward += 0.95 * phi - phi_prev

        return reward


    def step(self, action):
        phi_prev = self.step_phi
        (observation, reward, terminated, truncated, info) = MujocoHandReachEnv.step(self, action)
        phi = self.step_phi

        info['goalderivs'] = self.step_goalderivs
        info['phi_prev'] = phi_prev
        info['phi'] = phi
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

        # reward-design (inside vs. outside HER)
        IS_REWARD_REDESIGN = True
        if IS_REWARD_REDESIGN:
            reward = 0

             # reward slow down every derivative towards goal
            if np.all(self.step_goalderivs[::2] < 0) and np.all(self.step_goalderivs[1::2] > 0):
                reward = 1
            elif np.all(self.step_goalderivs > 0):
                reward = -1

            # vs.
            # reward some speed up towards goal ("get close faster")
            # if np.all(self.step_goalderivs < 0):
            #     reward = 1
            # # else:
            # if np.all(self.step_goalderivs > 0):
            #     reward = -1


        # if info['is_success']:
            # print('SUCCESS')
            # reward = 1 # success learning ("finish line") # irritates?!

        # goalprogress
        if self.goaldists and self.goaldists[0] > 0:
            self.goalprogress = (self.goaldists[0] - self.goaldists[-1]) / self.goaldists[0]
            self.goalprogress = max(0, self.goalprogress)
            info['goalprogress'] = self.goalprogress

        self.rewardsum += reward

        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(np.round(reward, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'rewardsum', str(np.round(self.rewardsum, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goalprogress', str(np.round(self.goalprogress, 2)))
            human_viewer.render()

        return observation, reward, terminated, truncated, info


    def reset(self, seed, options):
        print(self.goalprogress)
        print(self.rewardsum)
        self.outfile_goalprogresses.write('%s\n' % (self.goalprogress))
        self.outfile_goalprogresses.flush()

        self.subdists = []
        self.goaldists = []
        self.rewardsum = 0
        self.step_phi = np.zeros(ORDER_GOALDYNAMICS)
        self.step_goalderivs = np.zeros(ORDER_GOALDYNAMICS)
        return MujocoHandReachEnv.reset(self)
