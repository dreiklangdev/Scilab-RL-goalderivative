
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import BaseMujocoEnv
from ..le_base import base_practice_cfg

# TODO disable traj-halving, th-halving etc. in eval env

class BasePracticeEnv(BaseMujocoEnv):


    def __init__(self, cfg: base_practice_cfg):
        self.cfg: base_practice_cfg = cfg
        
        if self.cfg.MetaObservation.IS_ENABLED:
            self.obspace_total_dims = self.observation_space.shape[0] + self.cfg.PracticeSpace.d.shape[1] + 3
        else:
            self.obspace_total_dims = self.observation_space.shape[0]

        observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obspace_total_dims,), dtype='float64')
        goal_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=goal_space,
                achieved_goal=goal_space,
            )
        )

        # once
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goaldist_min_nld: float = np.inf
        self.ep_num_steps: int = 0
        self.desired_obs = self.cfg.PracticeSpace.d[2]
        self.goaldist_nld_personal_best: float = -1.0
        self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self._reset_episode()
        print('le-walker-2d initialized.')


    def extract_achieved_obs(superobs):
        raise NotImplementedError('inheriting env class must implement observing achieved goal from super obs')


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal_nld: np.ndarray, desired_goal_nld: np.ndarray, info
    ) -> float:

        reward = (achieved_goal_nld < desired_goal_nld)

        if np.isscalar(achieved_goal_nld[0]):
            # single live step (no replay)

            if self.cfg.GoalRewardThreshold.IS_NUDGING:
                if achieved_goal_nld[0] < self.goaldist_nld_personal_best:
                    print('personal record!', achieved_goal_nld[0])
                    self.goaldist_nld_personal_best = achieved_goal_nld[0]
                    reward = np.array([True])

        return reward.astype(np.float64)


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
        reward = float(reward[0])
        if reward:
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        terminated = False
        truncated = False

        # termination shaping? (manual vs autom.)
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        # achieved_obs_nld = self._normalize(self.extract_achieved_obs(obs['observation']), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        # dims_outside, = np.where(np.logical_or(achieved_obs_nld < 0, achieved_obs_nld > 1))
        # if len(dims_outside) > 0:
        #     # TODO automatic practice space
        #     print('outside practice space!',
        #           self.cfg.PracticeSpace.labels[dims_outside], achieved_obs_nld[dims_outside])
        #     terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
        #     reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        # TODO adaptive termination threshold? (halving again?)
        # if self.ep_goaldists_nld[-1] > self.ep_goaldists_nld[0] * 1.2:
        #     print('outside goal distance!', self.ep_goaldists_nld[-1])
        #     terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
        #     reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        window = 10
        if len(self.ep_goaldists_nld) >= window:
            is_converging = self.ep_goaldists_nld[-window] - self.ep_goaldists_nld[-1] < 0
            # is_converging = np.mean(np.gradient(self.ep_goaldists_nld[:window])) < 0
            if not is_converging:
                print('GOAL DIVERGENCE!', )
                terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
                reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        if self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            print('truncated.')
            info['success'] = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            truncated = True

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, truncated, info
        return result


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
            # brand new episode
            obs_init = super().reset_model()
            print('init_goaldistance', obs_init['achieved_goal'])
            self.desired_obs = self._new_desired_obs()
            self.last_ep_goaldist_min_nld = np.inf
            self.last_ep_rewards_mean = 0
            self.ep_traj_is_halved = False
            if self.goaldist_nld_personal_best < 0:
                self.goaldist_nld_personal_best = obs_init['achieved_goal']

            if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
                pad = self.cfg.GoalRewardThreshold.ADAPTION_PADDING
                # possibly traj-halved rewards mean
                if self.ep_rewards_mean < pad:
                    # bad episode before: towards init. goaldist.
                    self.ep_reward_threshold_nld += (obs_init['achieved_goal'] - obs_init['desired_goal']) / 2
                elif self.ep_rewards_mean > (1 - pad):
                    # good episode before: towards 0
                    self.ep_reward_threshold_nld -= (obs_init['achieved_goal'] - 0) / 2
                self.ep_reward_threshold_nld = max(self.ep_reward_threshold_nld, obs_init['achieved_goal'] * pad)
                self.ep_reward_threshold_nld = min(self.ep_reward_threshold_nld, obs_init['achieved_goal'] * (1 - pad))
                assert 0 <= self.ep_reward_threshold_nld <= obs_init['achieved_goal']

        self._reset_episode()
        return obs_init
    

    def _get_obs(self):
        superobs = super()._get_obs()

        achieved_obs_nld = self._normalize(self.extract_achieved_obs(superobs), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        desired_obs_nld = self._normalize(self.desired_obs, self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

        goaldiff_weighted = self.cfg.PracticeSpace.d[3] * np.array([achieved_obs_nld - desired_obs_nld])
        goaldist_nld = np.linalg.norm(goaldiff_weighted, axis=-1)

        metaobs = []
        metaobs.extend(self.desired_obs)
        if len(self.ep_goaldists_nld) > 1:
            goal_convergence = self.ep_goaldists_nld[-2] - self.ep_goaldists_nld[-1]
            metaobs.append(goal_convergence)
            is_converging = np.sign(self.ep_goaldists_nld[-2] - self.ep_goaldists_nld[-1])
            metaobs.append(is_converging)
        else:
            metaobs.extend([0,0])
        metaobs.append(goaldist_nld[0])

        obs = np.append(superobs, metaobs)
        self.ep_goaldists_nld.append(goaldist_nld[0])

        dictobs = dict(
                observation=obs,
                achieved_goal=goaldist_nld,
                desired_goal=self.ep_reward_threshold_nld,
            )
        
        return dictobs


    def _new_desired_obs(self):
        desired_obs_randomized = None

        match self.cfg.PracticeSpace.RandomGoalSampling.STRAT:
            case self.cfg.PracticeSpace.RandomGoalSampling.Strat.GENERALIST:
                desired_obs_randomized = np.random.uniform(self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

            case self.cfg.PracticeSpace.RandomGoalSampling.Strat.CONFORMIST:
                # https://en.wikipedia.org/wiki/Triangular_distribution
                desired_obs_randomized = np.random.triangular(self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[2], self.cfg.PracticeSpace.d[1])

            case self.cfg.PracticeSpace.RandomGoalSampling.Strat.SPECIALIST:
                desired_obs_randomized = self.cfg.PracticeSpace.d[2]

        desired_obs_randomized_weighted = self.cfg.PracticeSpace.d[3] * desired_obs_randomized + (1 - self.cfg.PracticeSpace.d[3]) * self.cfg.PracticeSpace.d[2]
        return desired_obs_randomized_weighted


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
        # self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * self.cfg.PracticeSpace.radius_normed
        self.ep_is_perfect = False
        print('obspace_total_dims', self.obspace_total_dims)
        print('desired_goal', self.desired_obs)
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
