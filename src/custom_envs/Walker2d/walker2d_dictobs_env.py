
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
        self.inaccuracy_threshold = 0.1

        self.ep_rewards_mean: float = 0
        self.ep_achieved_goal_mean: float = 0
        self.ep_inaccuracy_mean: float = 0
        self.ep_num_steps: int = 0


    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # torso-height, torso-angle, torso-velocity
        # achieved_goal = np.array((position[0], position[1], velocity[0]))
        achieved_goal = np.array((velocity[0]))
        # TODO manually find perfect desire! (how?)
        desired_goal = np.array((1.0))

        # min_goal = np.array((0.5))
        # max_goal = np.array((0.9))

        # manual normalization (obs fairness)
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        # achieved_goal = (achieved_goal - min_goal) / (max_goal - min_goal)
        # achieved_goal = np.array([achieved_goal])
        # achieved_goal[achieved_goal < 0] = 0
        # achieved_goal[achieved_goal > 1] = 1
        # desired_goal = (desired_goal - min_goal) / (max_goal - min_goal)
        # desired_goal = np.array([desired_goal])
        # desired_goal[desired_goal < 0] = 0
        # desired_goal[desired_goal > 1] = 1

        obs = dict(
                observation=observation,
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
            )

        return obs

    # TODO single-goal, then multi-goal
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        # closed-merged accuracy/distance vs. open-separate intervals
        # eg. "exactly" (better for episodic?) vs. "at-least" (tolerance better for infinite?)
        # TODO achieved_goal obs norm: weighting/normalization?
        # scalar accuracy training: too strict? ( no fine-tuning possible, but minimal/simple!)
        # distance increases with num_obs

        goal_diff = achieved_goal - desired_goal
        goal_diff = np.array([goal_diff])

        inaccuracy = np.linalg.norm(goal_diff, axis=-1)        
        self.ep_inaccuracy_mean = ((self.ep_num_steps * self.ep_inaccuracy_mean) + np.mean(inaccuracy)) / (self.ep_num_steps + 1)

        # threshold-sparse: binary vs. continuous
        # neg. rewards (episodic) vs. pos. rewards (continuous)
        # reward = max(self.init_inaccuracy - inaccuracy, 0)
        reward = (inaccuracy < self.inaccuracy_threshold)

        return np.float64(reward)


    def step(self, action):

        # reduce possible torque
        action = np.clip(action, -0.5, 0.5)

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
        # print(dir(self.data))
        
        n_contact = self.data.ncon

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_achieved_goal_mean = ((self.ep_num_steps * self.ep_achieved_goal_mean) + obs['achieved_goal']) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        # neutral termination (minimal?)
        # terminated = self.terminated or obs['achieved_goal'] < -0.3
        # accuracy termination?
        # problem: start may already be terminating state
        # inaccuracy = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=-1)

        height = obs['observation'][0]
        velocity = obs['observation'][1]

        terminated = 0

        # decrease search space
        if height < 0.8:
            print('height too low! ', height)
            terminated = 1

        if velocity < -0.3:
            print('negative velocity! ', velocity)
            terminated = 1

        if self.ep_num_steps > 150 and n_contact != 1:
            print('not only 1 foot! ', n_contact)
            terminated = 1

        if terminated:

            # adaptive goal?
            # if self.ep_rewards_mean > 0:
            #     self.goal_threshold = self.goal_threshold * 1.01

            # else:
            #     self.goal_threshold = self.goal_threshold * 0.99

            # self.goal_threshold = max(self.goal_threshold, 0.1)

            if True: # TODO move to info dict?
                print('ep_num_steps', self.ep_num_steps)
                print('ep_desired_goal', obs['desired_goal'])
                print('ep_achieved_goal_end', obs['achieved_goal'])
                print('ep_achieved_goal_mean', self.ep_achieved_goal_mean)
                print('ep_inaccuracy_mean', self.ep_inaccuracy_mean)
                print('ep_rewards_mean', self.ep_rewards_mean)
                print('\n')

            self.ep_num_steps = 0
            self.ep_achieved_goal_mean = 0
            self.ep_inaccuracy_mean = 0
            self.ep_rewards_mean = 0

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, False, info
        return result

    def reset_model(self):
        return super().reset_model()