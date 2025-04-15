
import numpy as np

from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from . import walker2d_dictobs_cfg as cfg


class BasePracticeEnv(Walker2dEnv, utils.EzPickle):


    def __init__(self):
        Walker2dEnv.__init__(self, exclude_current_positions_from_observation=False)

        orig_obspace = self.observation_space
        obspace = spaces.Box(-np.inf, np.inf, shape=(cfg.PracticeSpace.D.shape[1],), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        # once
        self.desired_goal = None
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goal_distance_min: float = np.inf

        # every ep
        self._reset_episode()
        print('le-walker-2d initialized.')


    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        goaldiff_weighted = cfg.PracticeSpace.D[3] * np.array([achieved_goal - desired_goal])
        # distance/accuracy (> at-least-only (needs control from both sides))
        goaldistance_normed = np.linalg.norm(goaldiff_weighted, axis=-1)
        if goaldistance_normed.shape[-1] == 1:
            # single step (no replay)
            self.ep_goal_distances_normed.append(goaldistance_normed[0])

        reward = (goaldistance_normed < self.ep_goal_reward_threshold_normed).astype(np.float64)
        # try reward if pos. goal convergence? (non-sparse)
        return reward


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        info = {}
        obs = self._get_obs()
        self.ep_obs_cur = obs

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        if reward:
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if cfg.GoalRewardThreshold.IS_ADAPTIVE:
            if reward and len(self.ep_goal_distances_normed) > 1:
                goaldistance_shrink = self.ep_goal_distances_normed[-2] - self.ep_goal_distances_normed[-1]
                goaldistance_shrink = max(0, goaldistance_shrink)
                self.ep_goal_reward_threshold_normed = self.ep_goal_distances_normed[-1] - goaldistance_shrink
                self.ep_goal_reward_threshold_normed = max(cfg.GoalRewardThreshold.MIN, self.ep_goal_reward_threshold_normed)
                self.ep_goal_reward_threshold_normed = min(cfg.GoalRewardThreshold.MAX_DEFAULT, self.ep_goal_reward_threshold_normed)
                if not self.ep_is_perfect:
                    if self.ep_goal_reward_threshold_normed == cfg.GoalRewardThreshold.MIN:
                        print('perfect goal zone reached! ', self.ep_goal_reward_threshold_normed / cfg.PracticeSpace.RADIUS)
                        self.ep_is_perfect = True
                    else:
                        print('adaptive threshold ratio ', self.ep_goal_reward_threshold_normed / cfg.PracticeSpace.RADIUS)

        terminated = False
        truncated = False

        # termination shaping?
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        dims_outside, = np.where(np.logical_or(obs['achieved_goal'] < 0, obs['achieved_goal'] > 1))
        
        if len(dims_outside) > 0:
            print('outside practice space!', cfg.PracticeSpace.LABELS[dims_outside], obs['achieved_goal'][dims_outside], sep=' ')
            terminated = cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
            reward = cfg.PracticeSpace.REWARD_IF_OUTSIDE

        if self.ep_num_steps > cfg.EPISODE_TRUNCATION_STEPS_MAX:
            print('truncated.')
            truncated = True

        if self.render_mode == "human":
            self.render()

        result = obs, float(reward), terminated, truncated, info
        return result
 

    def reset_model(self):
        obs_init = None

        if self.ep_obs_cur:
            # print(self.ep_goal_distances)
            print('ep_num_steps', self.ep_num_steps)
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_goal_distance_min_normed', min(self.ep_goal_distances_normed) / cfg.PracticeSpace.RADIUS)
            print('ep_goal_convergence_mean_per_step_normed', ((max(self.ep_goal_distances_normed) - min(self.ep_goal_distances_normed)) / self.ep_num_steps) / cfg.PracticeSpace.RADIUS)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            print('ep_goal_reward_threshold_normed', self.ep_goal_reward_threshold_normed / cfg.PracticeSpace.RADIUS)
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('ep_is_perfect', self.ep_is_perfect)
            print('\n')

        if self.ep_num_steps > 1:
            ep_goal_distance_min = min(self.ep_goal_distances_normed)

            # if IS_TRAJECTORY_HALVING and (self.ep_rewards_mean > self.last_ep_rewards_mean):
            if cfg.TrajectoryHalving.IS_ENABLED and (ep_goal_distance_min < self.last_ep_goal_distance_min):
                print('halving!')
                idx_halving = self._get_idx_for_trajectory_halving(cfg.TrajectoryHalving.STRAT)
                qpos, qvel = self.ep_states[idx_halving]
                qpos, qvel = self._add_noise(qpos, qvel)

                self.set_state(qpos, qvel)
                self.last_ep_rewards_mean = self.ep_rewards_mean
                self.last_ep_goal_distance_min = ep_goal_distance_min
                obs_init = self._get_obs()
                
        if not obs_init:
            # new goal
            self.desired_goal = self._get_goal()
            self.last_ep_goal_distance_min = np.inf
            self.last_ep_rewards_mean = 0
            obs_init = super().reset_model()

        self._reset_episode()
        return obs_init


    def _get_goal(self):
        if cfg.PracticeSpace.IS_RAND_GOAL_SAMPLING:
            # https://en.wikipedia.org/wiki/Triangular_distribution
            goal_randomized = np.random.triangular(cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[2], cfg.PracticeSpace.D[1])
            goal_randomized_weighted = cfg.PracticeSpace.D[3] * goal_randomized + (1 - cfg.PracticeSpace.D[3]) * cfg.PracticeSpace.D[2]
            return goal_randomized_weighted
        else:
            return cfg.PracticeSpace.D[2]


    def _add_noise(self, qpos, qvel):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        return qpos, qvel


    def _reset_episode(self):
        self.ep_rewards_mean: float = 0
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_obs_cur = None
        self.ep_goal_distances_normed = []
        self.ep_states = []
        self.ep_goal_reward_threshold_normed = cfg.GoalRewardThreshold.MAX_DEFAULT
        self.ep_is_perfect = False

        print('desired_goal ', self.desired_goal)


    def _get_idx_for_trajectory_halving(self, strat: cfg.TrajectoryHalving.Strat):
        idx_step = -1
        match strat:
            case cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(np.gradient(self.ep_goal_distances_normed))
            case cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goal_distances_normed)
        return idx_step


    def _normalize(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)
