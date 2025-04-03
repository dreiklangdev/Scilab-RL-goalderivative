
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
import numpy as np

THRESHOLD_ACCURACY = 0.1
# TODO include backwards? (similar enough?)
INTERVAL_SAMPLE_GOAL_VELOCITY = [0.1, 2.0]
INTERVAL_SAMPLE_GOAL_VELOCITY = [1.0, 1.0]

# TODO continue training with existent data?

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
        obspace = spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        print('walker-2d initialized.')
        self.ep_rewards_mean: float = 0
        self.ep_achieved_goal_mean: float = 0
        self.ep_num_steps: int = 0
        self.first_reward_state = None
        self.ep_first_reward_step: int = -1
        self.ep_goal_velocity: float = np.random.uniform(INTERVAL_SAMPLE_GOAL_VELOCITY[0], INTERVAL_SAMPLE_GOAL_VELOCITY[1])

    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()

        achieved_goal = np.array((velocity[0]))
        # achieved_goal = np.array((position[0], velocity[0]))
        desired_goal = np.array((self.ep_goal_velocity))

        min_goal = np.array((0.8, 0.5))
        max_goal = np.array((2, 0.9))


        obs = dict(
                observation=observation,
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
            )

        return obs

    def _normalize(self, achieved_goal, desired_goal):
        # manual normalization (obs fairness)
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        achieved_goal = (achieved_goal - min_goal) / (max_goal - min_goal)
        achieved_goal[achieved_goal < 0] = 0
        achieved_goal[achieved_goal > 1] = 1
        desired_goal = (desired_goal - min_goal) / (max_goal - min_goal)
        desired_goal[desired_goal < 0] = 0
        desired_goal[desired_goal > 1] = 1


    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        goal_diff = np.array([achieved_goal - desired_goal])
        # distance/accuracy (~min-max, != logical_and(), > at-least-only (needs control from both sides))
        accuracy = np.linalg.norm(goal_diff, axis=-1)
        reward = (accuracy < THRESHOLD_ACCURACY).astype(np.float64)
        return reward


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        info = {}
        obs = self._get_obs()
        
        n_contact = self.data.ncon

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_achieved_goal_mean = ((self.ep_num_steps * self.ep_achieved_goal_mean) + obs['achieved_goal']) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1
        
        # difficult
        terminated = False
        # easy
        truncated = False

        # decrease search/interaction space (find essential terminations)
        # imitation vs. direction (guidance, experience)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)

        height = obs['observation'][0]
        if height < 0.3:
            print('height too low! ', height)
            terminated = True
            reward = 0

        velocity = obs['observation'][1]
        if velocity < -0.3:
            print('negative velocity! ', velocity)
            terminated = True
            reward = 0

        mean_action = np.mean(np.abs(action))
        if mean_action < 0.001:
            print('non-activity!', mean_action)
            terminated = True
            reward = 0

        if self.ep_num_steps > 1500:
            print('truncated!')
            truncated = True

        # ---------

        if reward and self.ep_first_reward_step < 0:
            self.ep_first_reward_step = self.ep_num_steps

            if not self.first_reward_state:
                self.first_reward_state = (self.data.qpos.ravel().copy(), self.data.qvel.ravel().copy())

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, truncated, info
        return result
 

    def reset_model(self):
        obs_end = self._get_obs()
        print('ep_num_steps', self.ep_num_steps)
        print('ep_desired_goal', obs_end['desired_goal'])
        print('ep_achieved_goal_end', obs_end['achieved_goal'])
        print('ep_achieved_goal_mean', self.ep_achieved_goal_mean)
        print('ep_rewards_mean', self.ep_rewards_mean)
        print('first_reward_step', self.ep_first_reward_step)
        print('\n')

        if False and self.first_reward_state:
            # new promising init state
            # TODO need noise?
            self.set_state(self.first_reward_state[0], self.first_reward_state[1])
            obs_end = self._get_obs()

        else:
            obs_init = super().reset_model()

        self.ep_goal_velocity = np.random.uniform(INTERVAL_SAMPLE_GOAL_VELOCITY[0], INTERVAL_SAMPLE_GOAL_VELOCITY[1])
        self.ep_num_steps = 0
        self.ep_achieved_goal_mean = 0
        self.ep_rewards_mean = 0
        self.ep_first_reward_step = -1

        return obs_init