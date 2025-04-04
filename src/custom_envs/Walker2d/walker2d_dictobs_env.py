
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
import numpy as np

THRESHOLD_ACCURACY = 0.1 # wont be less

ADAPTIVE_ACCURACY_THRESHOLD = True
# TODO how to find perfect sparsity? (only manually?)
ADAPTIVE_ACCURACY_THRESHOLD_TARGET_REWARDS_MEAN = 0.1 # [0,1] REWARD SPARSITY HYPER PARAM - adapts threshold for specific rewards mean (hold constant difficulty level)
ADAPTIVE_ACCURACY_THRESHOLD_STEP = 0.1 # how fast it adapts (how well it holds the rewards mean (=sparsity)) 

# TODO include neg. backwards? (similar enough?)
# recognize impossible goal comps?
INTERVAL_SAMPLE_GOALS = np.array([
     # height, velocity
    [0.8, 0.3], # min
    [2.0, 2.0], # max
])


# https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
# https://gymnasium.farama.org/environments/mujoco/walker2d/
# src/custom_envs/maze/ant_env.py
# https://scilab-rl.github.io/Scilab-RL/wiki/Restore-a-saved-policy.html
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v4.py
# extend (vs wrapper)
class Walker2dDictObsEnv(Walker2dEnv, utils.EzPickle):


    def __init__(self):

        Walker2dEnv.__init__(self, exclude_current_positions_from_observation=True)

        orig_obspace = self.observation_space
        obspace = spaces.Box(-np.inf, np.inf, shape=(INTERVAL_SAMPLE_GOALS.shape[1],), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        self.ep_threshold_accuracy = THRESHOLD_ACCURACY
        self._reset_kpi()
        print('le-walker-2d initialized.')


    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()

        achieved_goal = np.array((position[0], velocity[0]))
        achieved_goal_norm, desired_goal_norm = self._normalize(achieved_goal, self.ep_desired_goal, INTERVAL_SAMPLE_GOALS[0], INTERVAL_SAMPLE_GOALS[1])
        # print(achieved_goal)

        obs = dict(
                observation=observation,
                achieved_goal=achieved_goal_norm,
                desired_goal=desired_goal_norm,
            )

        return obs
    

    def _normalize(self, achieved_goal, desired_goal, min_goal, max_goal):
        # manual normalization (obs fairness)
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        achieved_goal = (achieved_goal - min_goal) / (max_goal - min_goal)
        desired_goal = (desired_goal - min_goal) / (max_goal - min_goal)
        
        return achieved_goal, desired_goal


    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        goal_diff = np.array([achieved_goal - desired_goal])
        # distance/accuracy (~min-max, != logical_and(), > at-least-only (needs control from both sides))
        accuracy = np.linalg.norm(goal_diff, axis=-1)
        reward = (accuracy < self.ep_threshold_accuracy).astype(np.float64)
        return reward


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        info = {}
        obs = self._get_obs()     
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        if reward and self.ep_first_reward_step < 0:
            self.ep_first_reward_step = self.ep_num_steps

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if ADAPTIVE_ACCURACY_THRESHOLD:
            if self.ep_rewards_mean > ADAPTIVE_ACCURACY_THRESHOLD_TARGET_REWARDS_MEAN:
                self.ep_threshold_accuracy -= ADAPTIVE_ACCURACY_THRESHOLD_STEP
            else:
                self.ep_threshold_accuracy += ADAPTIVE_ACCURACY_THRESHOLD_STEP
            self.ep_threshold_accuracy = max(THRESHOLD_ACCURACY, self.ep_threshold_accuracy)

        # too difficult
        terminated = False
        # too easy
        truncated = False

        # faster learning: decrease search/interaction space (find terminations)
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        height = obs['observation'][0]
        if height < 0.7:
            print('height too low! ', height)
            terminated = True
            reward = 0

        angle = obs['observation'][1]
        if not (-1 < angle < 1):
            print('illegal angle! ', angle)
            terminated = True
            reward = 0

        velocity = obs['observation'][8]
        # if velocity < -0.3:
        #     print('negative velocity! ', velocity)
        #     terminated = True
        #     reward = 0
        if self.ep_num_steps > 300 and velocity < 0.3:
            print('not forward!', velocity)
            terminated = True
            reward = 0


        mean_velocity_all = np.mean(np.abs(obs['observation']))
        if mean_velocity_all < 0.2:
            print('standing still!', mean_velocity_all)
            terminated = True
            reward = 0

        n_contact = self.data.ncon
        # if self.ep_num_steps > 150 and n_contact == 0:
        #     print('jumping!')
        #     terminated = True
        #     reward = 0

        if self.ep_num_steps > 1000:
            print('truncated!')
            truncated = True

        # ---------

        if self.render_mode == "human":
            self.render()

        self.ep_obs_cur = obs
        result = obs, reward, terminated, truncated, info
        return result
 

    def reset_model(self):
        if self.ep_obs_cur:
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_num_steps', self.ep_num_steps)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('ep_threshold_accuracy', self.ep_threshold_accuracy)
            print('\n')

        ep_obs_init = super().reset_model()
        self._reset_kpi()
        return ep_obs_init
    

    def _reset_kpi(self):
        self.ep_rewards_mean: float = 0
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_desired_goal = np.random.uniform(INTERVAL_SAMPLE_GOALS[0], INTERVAL_SAMPLE_GOALS[1])
        self.ep_obs_cur = None