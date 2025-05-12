
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import BaseMujocoEnv
from ..le_base import base_practice_cfg

# TODO disable traj-halving, th-halving etc. in eval env

class BasePracticeEnv(BaseMujocoEnv):


    def __init__(self, cfg: base_practice_cfg):
        self.cfg: base_practice_cfg = cfg
        
        obspace_total_dims = self.observation_space.shape[0] # super
        obspace_total_dims += self.cfg.PracticeSpace.d.shape[0] # achieved

        if self.cfg.MetaObservation.IS_ENABLED:
            obspace_total_dims += self.cfg.PracticeSpace.d.shape[1] + 3 # meta

        observation_space = spaces.Box(-np.inf, np.inf, shape=(obspace_total_dims,), dtype='float64')
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
        self.last_ep_goaldist_min: float = np.inf
        self.ep_num_steps: int = 0
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT
        self.desired_obs = self.cfg.PracticeSpace.d[2]
        self.goaldist_personal_best: float = -1.0
        self.total_full_episodes = 0

        self._reset_episode()
        print('le-walker-2d initialized.')
        print('observation_space', observation_space)
        print('goal_space', goal_space)


    def extract_practiced_obs(superobs):
        raise NotImplementedError('inheriting env class must implement observing practiced obs from super obs')


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        reward = (achieved_goal < desired_goal)

        if achieved_goal.ndim == 0:
            # single live step (no replay)
            
            if self.cfg.GoalRewardThreshold.IS_NUDGING:
                if achieved_goal < self.goaldist_personal_best:
                    print('personal record!', achieved_goal)
                    self.goaldist_personal_best = achieved_goal
                    reward = np.bool_(True)

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

        # ~ grace time?
        # 20k /home/t14/Documents/tuhh/dsf/Scilab-RL/data/785d4a5/le-walker2d-v4/13-05-49/rl_model_finished
        # 100k /home/t14/Documents/tuhh/dsf/Scilab-RL/data/785d4a5/le-walker2d-v4/13-05-49_restored/rl_model_finished
        if self.ep_num_steps >= 100 and self.ep_num_steps % 100 == 0:
            # allow nudging every n steps again
            print('reset personal best.')
            self.goaldist_personal_best = np.inf


        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        if reward:
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        terminated = False
        truncated = False

        # practice pre-knowledge
        # space-constraint
        # always manual only? (direction (guidance, experience, coaching))
        # the more, the better?
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO autom. practice space bounds: input := min-max. (sampled?), output:= rewards_mean
        #   (bounds/constrained opt./adapt.: bayes?) (unconstrained opt./adapt.: gradient descent?)
        if self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE:
            practiced_obs = self._normalize(self.extract_practiced_obs(obs['observation']), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
            dims_outside, = np.where(np.logical_or(practiced_obs < 0, practiced_obs > 1))
            if len(dims_outside) > 0:
                print('OUTSIDE PRACTICE SPACE!', self.cfg.PracticeSpace.labels[dims_outside], practiced_obs[dims_outside])
                terminated = True                    
                # dont neutralize already pos. eps.
                if not self.ep_rewards_mean and not reward:
                    reward = self.cfg.PracticeSpace.REWARD_ON_TERMINATE
                

        # possibly viable for envs without significant practice pre-knowledge (eg. no space-constraints)?
        # time-constraint
        # 100k, disabled, space:   2 steps, confident, efficient, jumpy /home/t14/Documents/tuhh/dsf/Scilab-RL/data/cee5d5e/le-walker2d-v4/15-09-46/rl_model_finished
        # 200k, disabled, space:   2-3 steps /home/t14/Documents/tuhh/dsf/Scilab-RL/data/c3315cd/le-walker2d-v4/16-01-27/rl_model_finished  
        # 100k, enabled:    no step, less efficient /home/t14/Documents/tuhh/dsf/Scilab-RL/data/cee5d5e/le-walker2d-v4/14-53-42/rl_model_finished
        # 200k, enabled:    1-2 steps /home/t14/Documents/tuhh/dsf/Scilab-RL/data/c3315cd/le-walker2d-v4/15-30-10/rl_model_finished
        # 100k, noSpace:    0.5 step /home/t14/Documents/tuhh/dsf/Scilab-RL/data/c3315cd/le-walker2d-v4/21-06-14/rl_model_finished
        # 100k, noSpace, noNudge:   -1 step /home/t14/Documents/tuhh/dsf/Scilab-RL/data/c3315cd/le-walker2d-v4/21-35-34/rl_model_finished
        # 200k, noSpace, nudge:     1 step /home/t14/Documents/tuhh/dsf/Scilab-RL/data/c3315cd/le-walker2d-v4/21-06-14/rl_model_finished
        # 200k, noSpace, nudge, 2d:     0-0.5 steps /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fea1c75/le-walker2d-v4/22-55-10/rl_model_finished
        # 200k, noSpace, nudge, 2d, penalty:    1.5 step reliably forward by tumbling /home/t14/Documents/tuhh/dsf/Scilab-RL/data/fea1c75/le-walker2d-v4/23-26-55/rl_model_finished
        # 200k, space, nudge, 2d, penalty:  3-4 steps confident, reliably forward but collapsing walk /home/t14/Documents/tuhh/dsf/Scilab-RL/data/7d45aa7/le-walker2d-v4/00-10-10/rl_model_finished
        if self.cfg.PracticeTime.IS_TERMINATE_ON_GRACE_STEPS_DIVERGENCE:
            grace_steps = self.cfg.PracticeTime.GRACE_STEPS
            if len(self.ep_goaldists) >= grace_steps:
                is_goal_reached = obs['achieved_goal'] < obs['desired_goal']
                is_goal_converging = self.ep_goaldists[-grace_steps] - self.ep_goaldists[-1] < 0
                # is_converging = np.median(np.gradient(self.ep_goaldists_nld[:GRACE_STEPS])) < 0
                if not is_goal_reached and not is_goal_converging:
                    print('NO GOAL CONVERGENCE AFTER GRACE STEPS!', grace_steps)                
                    terminated = True
                    # dont neutralize already pos. eps.
                    if not self.ep_rewards_mean and not reward:
                        reward = self.cfg.PracticeTime.REWARD_ON_TERMINATE

        # avoid & seek
        # 20k   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/785d4a5/le-walker2d-v4/12-06-31/rl_model_finished
        

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

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
            print('ep_goal_desired', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved', self.ep_obs_cur['achieved_goal'])
            print('ep_goaldist_min', min(self.ep_goaldists))
            print('ep_goaldist_mean', np.mean(self.ep_goaldists))
            print('ep_goaldist_max', max(self.ep_goaldists))
            print('ep_reward_threshold', self.ep_reward_threshold)
            print('ep_traj_is_halved', self.ep_traj_is_halved)
            print('ep_rewards_mean', self.ep_rewards_mean)
            if self.ep_rewards_mean == 0:
                print('WARNING: zero-sum-ep. => wasted ep.')
            print('\n')

        if self.ep_num_steps > 1:
            ep_goaldist_min = min(self.ep_goaldists)
            if ep_goaldist_min < self.last_ep_goaldist_min:
                if self.cfg.TrajectoryHalving.IS_ENABLED:
                    idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
                    print('halving!', idx_halving)
                    qpos, qvel = self.ep_states[idx_halving]
                    qpos, qvel = self._add_noise(qpos, qvel)

                    self.set_state(qpos, qvel)
                    self.ep_traj_is_halved = True
                    self.last_ep_rewards_mean = self.ep_rewards_mean
                    self.last_ep_goaldist_min = ep_goaldist_min
                    obs_init = self._get_obs()

        if not obs_init:
            # brand new episode
            obs_init = super().reset_model()
            print('init_goaldistance', obs_init['achieved_goal'])
            self.desired_obs = self._get_desired_obs()
            self.last_ep_goaldist_min = np.inf
            self.last_ep_rewards_mean = 0
            self.ep_traj_is_halved = False
            # self.ep_reward_threshold_nld = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * obs_init['achieved_goal']

            if self.goaldist_personal_best < 0:
                self.goaldist_personal_best = obs_init['achieved_goal']

            if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
                self.ep_reward_threshold = (1 - self.ep_rewards_mean) * (obs_init['achieved_goal'])

            self.total_full_episodes += 1
            print('total_full_episodes', self.total_full_episodes)

        self._reset_episode()
        return obs_init
    

    def _get_obs(self):
        obs = []
        superobs = super()._get_obs()
        obs.extend(superobs)

        achieved_obs = self._normalize(self.extract_practiced_obs(superobs), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        obs.extend(achieved_obs)
        
        desired_obs = self._normalize(self.desired_obs, self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

        goaldiff_weighted = self.cfg.PracticeSpace.d[3] * np.array(achieved_obs - desired_obs)
        goaldist = np.linalg.norm(goaldiff_weighted, axis=-1)

        if self.cfg.MetaObservation.IS_ENABLED:
            metaobs = []
            metaobs.extend(desired_obs)
            if len(self.ep_goaldists) > 1:
                goal_convergence = self.ep_goaldists[-2] - self.ep_goaldists[-1]
                metaobs.append(goal_convergence)
                is_converging = np.sign(self.ep_goaldists[-2] - self.ep_goaldists[-1])
                metaobs.append(is_converging)
            else:
                metaobs.extend([0,0])
            metaobs.append(goaldist)
            obs.extend(metaobs)

        self.ep_goaldists.append(goaldist)

        dictobs = dict(
                observation=obs,
                achieved_goal=goaldist,
                desired_goal=self.ep_reward_threshold,
            )

        return dictobs


    def _get_desired_obs(self):
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
        self.ep_rewards_mean: float = -1
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_obs_cur = None
        self.ep_goaldists = []
        self.ep_states = []
        self.ep_is_perfect = False
        print('desired_obs', self.desired_obs)
        # reset nudging every ep.?
        # 20k /home/t14/Documents/tuhh/dsf/Scilab-RL/data/785d4a5/le-walker2d-v4/12-42-29/rl_model_finished
        # 50k: not good, no fall, but no walk /home/t14/Documents/tuhh/dsf/Scilab-RL/data/785d4a5/le-walker2d-v4/12-42-29_restored/rl_model_finished
        # self.goaldist_personal_best = np.inf
        print('goaldist_personal_best', self.goaldist_personal_best)

    def _get_idx_for_trajectory_halving(self, strat):
        idx_step = -1
        match strat:
            case self.cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case self.cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(np.gradient(self.ep_goaldists))
            case self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goaldists)
        return idx_step


    def _normalize(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)
