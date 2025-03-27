
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
import gymnasium as gym
import numpy as np

# gym.pprint_registry()


# TODO observations: box to dict
# TODO HER
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
        obspace = spaces.Box(-np.inf, np.inf, shape=(2,), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        print('walker-2d initialized.')
        self.goal_velocity_range = (0.3, 0.5)
        self.ep_rewards_mean: float = 0
        self.ep_velocity_mean: float = 0
        self.ep_num_steps: int = 0


    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()

        min_z, max_z = self._healthy_z_range

        # achieved_goal = np.concatenate(([position[1]], [velocity[0]])).ravel()
        achieved_goal = np.array((velocity[0], position[0]))
        desired_goal = self.goal_velocity_range[0]

        obs = dict(
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
                observation=observation,
            )

        return obs


    # only needed with HER

    # TODO single-goal, then multi-goal
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        # distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # success = (distance <= self.distance_threshold).astype(np.float64)
        

        # distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        reward = 1

        velocity = achieved_goal[0]
        height = achieved_goal[1]

        # sparse reward(binary 0,1) vs. sparse penalty/reward(-1,1)
        # rewards[velocity >= self.goal_velocity_range[0] and velocity <= self.goal_velocity_range[1]] = 1

        # neutralize(0) vs. punish(-1)

        if velocity < self.goal_velocity_range[0] or velocity > self.goal_velocity_range[1]:
            reward = 0
        if height < 0.8:
            reward = 0

        # success = (distance <= self.distance_threshold).astype(np.float64)
        # success = np.array(velocity > self.goal_velocity_range[0] and velocity < self.goal_velocity_range[1]).astype(np.float64)
        # penalty / punishment (may lead to non-action? ("fear of action"))
        # (0,1) vs. (-1,1) vs. (-k,1) vs. (-1,k)

        return np.float64(reward)


    def step(self, action):
        # x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        # x_position_after = self.data.qpos[0]
        # x_velocity = (x_position_after - x_position_before) / self.dt

        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        # healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        # costs = ctrl_cost

        # info = {
        #     "x_position": x_position_after,
        #     "x_velocity": x_velocity,
        # }

        info = {}
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        velocity = obs['achieved_goal'][0]
        height = obs['achieved_goal'][1]

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_velocity_mean = ((self.ep_num_steps * self.ep_velocity_mean) + velocity) / (self.ep_num_steps + 1)
        self.ep_num_steps = self.ep_num_steps + 1

        # neutral termination (minimal?)
        # terminated = self.terminated or obs['achieved_goal'] < -0.3
        terminated = height < 0.3


        if terminated:

            # adaptive goal?
            # if self.ep_rewards_mean > 0:
            #     self.goal_threshold = self.goal_threshold * 1.01

            # else:
            #     self.goal_threshold = self.goal_threshold * 0.99

            # self.goal_threshold = max(self.goal_threshold, 0.1)


            print('termination!')
            print('termination: ep_rewards_mean', self.ep_rewards_mean)
            print('termination: ep_num_steps', self.ep_num_steps)
            print('goal_velocity_range', self.goal_velocity_range)
            print('ep_velocity_mean', self.ep_velocity_mean)

            self.ep_rewards_mean = 0
            self.ep_num_steps = 0

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, False, info
        return result
