
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

# v3.0 - refactored, very sparse
#   200k, reset threshold per ep.   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/19-50-45_restored_restored/rl_model_finished
#   300k, neg. reward       /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/08-07-14_restored_restored_restored_restored/rl_model_finished
#   400k, adaptive[0,1]     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/09-03-13_restored/rl_model_finished
#   400k, adaptive[0,0.5]   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/09-57-22_restored_restored/rl_model_finished
#   120k, noAdapt   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/11-06-36_restored/rl_model_finished

# v4.0 - strong weight diff.
#   120k, bidir. adaptive   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2a19989/le-walker2d-v4/19-31-37_restored_restored_restored/rl_model_finished

# v5 - shrinking adapt.
#   1.  200k, minmax[0,1]   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/13-16-01/rl_model_finished
#                        /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/14-12-21/rl_model_finished
#   2.  400k,               /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/14-12-21_restored/rl_model_finished
#   3.  400k, strongweight, halfAtHighestConv   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/15-38-32/rl_model_finished
#   4.  400k,               halfAtHighestConv,initShrink    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/18-16-05/rl_model_finished
#   5.  400k,               halfAtLowestDist    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/a38f6f0/le-walker2d-v4/17-25-07/rl_model_finished
#   6.  400k, 0.2           noAdaptive, no trajhalv     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/592a508/le-walker2d-v4/21-02-44/rl_model_finished
#   7.  400k, 0.2           noAdaptive, halfAtLowestDist    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/592a508/le-walker2d-v4/20-06-07/rl_model_finished
#   8.  400k, minmax[0,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8d73cee/le-walker2d-v4/08-38-24/rl_model_finished              
#   9.  400k, minmax[0.05,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/638089b/le-walker2d-v4/09-46-08/rl_model_finished             
#   10.  400k, minmax[0.01,0.2], strongweight, halfAtLowestDist     /home/t14/Documents/tuhh/dsf/Scilab-RL/data/638089b/le-walker2d-v4/10-49-20/rl_model_finished             
# adaptive threshold (high max.) results in overall jerky, unsecure movements (unclear rewarding!), possibly better for universal training (rand. goals)? - no!
# adaptive threshold (high max.) also leads to large actor-/critic-losses
# need strongly different goal weighting!
# great improvement with trajHalv
# adapt. threshold: "perfecting" inside goal zone ("finetune", sufficient goal zone vs. perfect goal zone)

# v6 - random goals, no adapt
#   1.  400k, randomAllDimsNoWeights,Th0.2    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/21-57-09/rl_model_finished
#   2.  400k, randomWeightedDims,Th0.2        /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/23-35-09/rl_model_finished
#   3.  400k, randomWeightedDims,adaptiveTh   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/29743f1/le-walker2d-v4/00-34-57/rl_model_finished
# random: slower, but generalizing (single dim, ie. velocity)

# best: v7    0.2, noAdaptive, th-halfAtLowestDist
# best: v9    minmax[0.05,0.2], th-halfAtLowestDist

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
# TODO create (abstract?) base class
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
                self.ep_goal_reward_threshold_normed = min(cfg.GoalRewardThreshold.MAX, self.ep_goal_reward_threshold_normed)
                if not self.ep_is_perfect and self.ep_goal_reward_threshold_normed == cfg.GoalRewardThreshold.MIN:
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
        self.ep_goal_reward_threshold_normed = cfg.GoalRewardThreshold.MAX
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
