
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import BaseMujocoEnv
from ..le_base import base_practice_cfg


class BasePracticeEnv(BaseMujocoEnv):


    def __init__(self, cfg: base_practice_cfg):
        self.cfg: base_practice_cfg = cfg

        if self.cfg.MetaObservation.IS_ENABLED:
            self.obspace_total_dims = self.observation_space.shape[0] + self.cfg.PracticeSpace.d.shape[1] + 3
        else:
            self.obspace_total_dims = self.observation_space.shape[0]

        observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obspace_total_dims,), dtype='float64')
        desired_obs_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype='float64')
        achieved_obs_space = desired_obs_space

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=desired_obs_space,
                achieved_goal=achieved_obs_space,
            )
        )

        # once
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goaldist_min_nld: float = np.inf
        self.ep_num_steps: int = 0
        self.desired_goal = self.cfg.PracticeSpace.d[2]
        self.goaldist_nld_personal_best = np.inf
        self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self._reset_episode()
        print('le-walker-2d initialized.')


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        info = {}
        info['success'] = False
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.ep_states.append((qpos, qvel))

        obs = self._get_obs()
        self.ep_obs_cur = obs

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        reward = float(reward)

        if reward:
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
            if reward and len(self.ep_goaldists_nld) > 1:
                goaldistance_shrink = self.ep_goaldists_nld[-2] - self.ep_goaldists_nld[-1]
                goaldistance_shrink = max(0, goaldistance_shrink)
                self.ep_reward_threshold_nld = self.ep_goaldists_nld[-1] - goaldistance_shrink
                self.ep_reward_threshold_nld = max(self.cfg.GoalRewardThreshold.MIN_FRAC, self.ep_reward_threshold_nld)
                self.ep_reward_threshold_nld = min(self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT, self.ep_reward_threshold_nld)
                if not self.ep_is_perfect:
                    if self.ep_reward_threshold_nld == self.cfg.GoalRewardThreshold.MIN_FRAC:
                        print('perfect goaldist zone reached!', self.ep_reward_threshold_nld)
                        self.ep_is_perfect = True
                    else:
                        print('adaptive threshold ratio', self.ep_reward_threshold_nld)

        terminated = False
        truncated = False

        # termination shaping? (manual vs autom.)
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        achieved_obs_nld = self._normalize(self.get_achieved_goal(obs['observation']), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        dims_outside, = np.where(np.logical_or(achieved_obs_nld < 0, achieved_obs_nld > 1))
        if len(dims_outside) > 0:
            # TODO automatic practice space
            print('outside practice space!',
                  self.cfg.PracticeSpace.labels[dims_outside], achieved_obs_nld[dims_outside])
            terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
            reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        # TODO adaptive termination threshold?
        # if self.ep_goaldists_normed[-1] > self.ep_goaldists_normed[0] * 1.2:
        #     print('outside goal distance!', self.ep_goaldists_normed[-1])
        #     terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
        #     reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        if self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            print('truncated.')
            info['success'] = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            truncated = True

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, truncated, info
        return result
    

    def get_achieved_goal(obs):
        raise NotImplementedError('inheriting env class must implement extract achieved goal from obs')


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal_nld: np.ndarray, desired_goal_nld: np.ndarray, info
    ) -> float:

        goaldiff_weighted = self.cfg.PracticeSpace.d[3] * np.array([achieved_goal_nld - desired_goal_nld])
        goaldist_nld = np.linalg.norm(goaldiff_weighted, axis=-1)[0]
        reward = (goaldist_nld < self.ep_reward_threshold_nld)

        if np.isscalar(goaldist_nld):
            # live step (no replay)
            if self.cfg.GoalRewardThreshold.IS_NUDGING:
                if goaldist_nld < self.goaldist_nld_personal_best:
                    print('personal record!', goaldist_nld)
                    self.goaldist_nld_personal_best = goaldist_nld
                    # TODO adaptive threshold on record breaking? (incl. min threshold?)
                    self.ep_reward_threshold_nld = goaldist_nld
                    reward = np.bool_(True)

        return reward.astype(np.float64)
    

    def reset_model(self):
        obs_init = None

        if self.ep_obs_cur:
            print('ep_num_steps', self.ep_num_steps)
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_goal_desired_nld', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_nld', self.ep_obs_cur['achieved_goal'])
            print('ep_goaldist_min_nld', min(self.ep_goaldists_nld))
            print('ep_goaldist_mean_nld', np.mean(self.ep_goaldists_nld))
            print('ep_goaldist_max_nld', max(self.ep_goaldists_nld))
            print('ep_reward_threshold_nld', self.ep_reward_threshold_nld)
            print('ep_traj_is_halved', self.ep_traj_is_halved)
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('\n')

        if self.ep_num_steps > 1:
            ep_goaldist_min_nld = min(self.ep_goaldists_nld)

            if self.cfg.TrajectoryHalving.IS_ENABLED and (ep_goaldist_min_nld < self.last_ep_goaldist_min_nld):
                idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
                print('halving!', idx_halving)
                qpos, qvel = self.ep_states[idx_halving]
                qpos, qvel = self._add_noise(qpos, qvel)

                self.set_state(qpos, qvel)
                self.ep_traj_is_halved = True
                self.last_ep_rewards_mean = self.ep_rewards_mean
                self.last_ep_goaldist_min_nld = ep_goaldist_min_nld
                obs_init = self._get_obs()

        if not obs_init:
            obs_init = super().reset_model()
            self.desired_goal = self._new_desired_goal()
            self.last_ep_goaldist_min_nld = np.inf
            self.last_ep_rewards_mean = 0
            self.ep_traj_is_halved = False

        self._reset_episode()
        return obs_init
    

    def _get_obs(self):
        superobs = super()._get_obs()

        achieved_goal_nld = self._normalize(self.get_achieved_goal(superobs), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        desired_goal_nld = self._normalize(self.desired_goal, self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

        goaldiff_weighted = self.cfg.PracticeSpace.d[3] * np.array([achieved_goal_nld - desired_goal_nld])
        goaldist_nld = np.linalg.norm(goaldiff_weighted, axis=-1)[0]
        self.ep_goaldists_nld.append(goaldist_nld)

        metaobs = []
        metaobs.extend(self.desired_goal)
        if len(self.ep_goaldists_nld) > 1:
            goal_convergence = self.ep_goaldists_nld[-2] - self.ep_goaldists_nld[-1]
            metaobs.append(goal_convergence)
            is_converging = np.sign(self.ep_goaldists_nld[-2] - self.ep_goaldists_nld[-1])
            metaobs.append(is_converging)
        else:
            metaobs.extend([0,0])
        metaobs.append(goaldist_nld)

        obs = np.append(superobs, metaobs)

        dictobs = dict(
                observation=obs,
                achieved_goal=achieved_goal_nld,
                desired_goal=desired_goal_nld,
            )

        return dictobs


    def _new_desired_goal(self):
        desired_goal_randomized = None

        match self.cfg.PracticeSpace.RandomSampling.STRAT:
            case self.cfg.PracticeSpace.RandomSampling.Strat.GENERALIST:
                desired_goal_randomized = np.random.uniform(self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

            case self.cfg.PracticeSpace.RandomSampling.Strat.CONFORMIST:
                # https://en.wikipedia.org/wiki/Triangular_distribution
                desired_goal_randomized = np.random.triangular(self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[2], self.cfg.PracticeSpace.d[1])

            case self.cfg.PracticeSpace.RandomSampling.Strat.SPECIALIST:
                desired_goal_randomized = self.cfg.PracticeSpace.d[2]

        desired_goal_weighted = self.cfg.PracticeSpace.d[3] * desired_goal_randomized + (1 - self.cfg.PracticeSpace.d[3]) * self.cfg.PracticeSpace.d[2]
        return desired_goal_weighted


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
        self.ep_goaldists_nld = []
        self.ep_states = []
        # self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT
        self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * self.cfg.PracticeSpace.radius_normed
        self.ep_is_perfect = False
        print('obspace_total_dims', self.obspace_total_dims)
        print('desired_goal', self.desired_goal)
        print('goaldist_nld_personal_best', self.goaldist_nld_personal_best)


    def _get_idx_for_trajectory_halving(self, strat):
        idx_step = -1
        match strat:
            case self.cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case self.cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(np.gradient(self.ep_goaldists_nld))
            case self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goaldists_nld)
        return idx_step


    def _normalize(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)
