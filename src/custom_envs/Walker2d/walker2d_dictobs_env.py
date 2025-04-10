
import numpy as np

from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from . import walker2d_dictobs_cfg as cfg


# ref https://www.youtube.com/watch?v=irkXnpZP89s
# records
# v1.0    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aab0ae/le-walker2d-v4/17-58-22/rl_model_finished
#   400k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43/rl_model_finished
#   500k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43_restored/rl_model_finished
#   400k, velo2, no practicemode    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/48779a8/le-walker2d-v4/00-14-47/rl_model_finished

# v2.0 - distance-dim., soft-/hard-terms
#   400k, soft-hard-terms   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/15-35-01/rl_model_finished
#   400k, soft-term-only    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/16-28-54/rl_model_finished
#   400k, hard-term-only    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fbbe332/le-walker2d-v4/17-15-35/rl_model_finished

# forming: goal + termination ("coaching")
# reward-trickling ("breadcrumbing")

# glossar
# GOALSPACE_DESIRED := goalstate -+ threshold
# (GOAL-)STATE_ACHIEVED := current state of step
# PRACTICE-/TRAINSPACE := all reasonable states to learn from

# TODO goal: time-dim. vs. infinite (non-episodic), stand-up? 

# https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
# https://gymnasium.farama.org/environments/mujoco/walker2d/
# src/custom_envs/maze/ant_env.py
# https://scilab-rl.github.io/Scilab-RL/wiki/Restore-a-saved-policy.html
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v4.py
# extend (vs wrapper)
class Walker2dDictObsEnv(Walker2dEnv, utils.EzPickle):


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
        self.ep_goal_reward_threshold = cfg.GoalRewardThreshold.MIN
        self.desired_goal = None
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goal_distance_min: float = np.inf

        # every ep
        self._reset_episode()
        print('le-walker-2d initialized.')


    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = np.clip(self.data.qvel.flat.copy(), -10, 10)
        self.ep_states.append((qpos, qvel))

        observation = np.concatenate((qpos, qvel)).ravel()
        distance, height, velocity, angle = qpos[0], qpos[1], qvel[0], qpos[2]
        n_contact_after = self.data.ncon if self.ep_num_steps > 300 else 1
        angle_thigh = max(qpos[3], qpos[6])
        is_moving_forward = velocity > 0.3 if self.ep_num_steps > 300 else 1

        achieved_goal = np.array((height, velocity, angle, angle_thigh, is_moving_forward))
        achieved_goal_norm = self._normalize(achieved_goal, cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[1])
        desired_goal_norm = self._normalize(self.desired_goal, cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[1])

        obs = dict(
                observation=observation,
                achieved_goal=achieved_goal_norm,
                desired_goal=desired_goal_norm,
            )

        return obs
    

    def _normalize(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)


    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        goaldiff_weighted = cfg.PracticeSpace.D[3] * np.array([achieved_goal - desired_goal])
        # distance/accuracy (> at-least-only (needs control from both sides))
        goaldistance = np.linalg.norm(goaldiff_weighted, axis=-1)
        if goaldistance.shape[-1] == 1:
            # single step (no replay)
            self.ep_goal_distances.append(goaldistance[0])

        reward = (goaldistance < self.ep_goal_reward_threshold).astype(np.float64)
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
            if self.ep_rewards_mean > cfg.GoalRewardThreshold.ADAPTIVE_REWARD_MEAN:
                self.ep_goal_reward_threshold -= cfg.GoalRewardThreshold.ADAPTIVE_REWARD_CHANGE
            else:
                self.ep_goal_reward_threshold += cfg.GoalRewardThreshold.ADAPTIVE_REWARD_CHANGE
            self.ep_goal_reward_threshold = max(cfg.GoalRewardThreshold.MIN, self.ep_goal_reward_threshold)
            self.ep_goal_reward_threshold = min(cfg.GoalRewardThreshold.MAX, self.ep_goal_reward_threshold)

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
            terminated = cfg.PracticeSpace.IS_TERMINATION_ON_LEAVING
            reward = 0
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
            print('ep_goal_distance_min', min(self.ep_goal_distances))
            print('ep_goal_convergence_per_step', (max(self.ep_goal_distances) - min(self.ep_goal_distances)) / self.ep_num_steps)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            print('ep_goal_reward_threshold', self.ep_goal_reward_threshold)
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('\n')

        if self.ep_num_steps > 1:
            ep_goal_distance_min = min(self.ep_goal_distances)

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
            return np.random.triangular(cfg.PracticeSpace.D[0], cfg.PracticeSpace.D[2], cfg.PracticeSpace.D[1])
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
        self.ep_goal_distances = []
        self.ep_states = []
        if cfg.GoalRewardThreshold.IS_RESET_PER_EPISODE:
            self.ep_goal_reward_threshold = cfg.GoalRewardThreshold.MIN
        print('desired_goal ', self.desired_goal)


    def _get_idx_for_trajectory_halving(self, strat: cfg.TrajectoryHalving.Strat):
        idx_step = -1
        match strat:
            case cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(np.gradient(self.ep_goal_distances))
            case cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goal_distances)
        return idx_step
