from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
import numpy as np

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

# every training must be inside practice space (reach, hold, recover, etc.)
# what if contradictory goals?
# practice/train space
# = same as all(!) reasonable/possible/unconstrained states (incl. start-state) (transfer-learning?)
# TODO how to find perfect practice space? (watch/oversee (at the start)! eg. too strict vs. too lax, also learn to (slightly) recover?)
# TODO how to find (near-)impossible goals? (eg. distance vs. velocity)
# intervals
#   smaller => faster, focussed
#   larger => slower, more universal
# generally: the more goal dims., the better? ("more experienced coach")
# TODO goal analysis (eg. most failed dim.) on eval
# sample int (float), if int (float)?
PRACTICE_SPACE_LABELS = np.array([
    'height',   'velocity',  'angle',   'contact_after', 'angle_thigh',  'is_moving_forward',
])
PRACTICE_SPACE = np.array([
    [0.8,        -2.0,       -1.0,       1,              -2.0,           1],     # min
    [2.0,        3.0,        1.5,        3,              2.0,            1.1],   # max
    [1.1,        2.0,        0.5,        1,              0,              1],     # mode
    [1.0,        2.0,        0.5,        0.5,            0.0,            1.0]    # weight
])
PRACTICE_SPACE_DIAMETER = np.linalg.norm(PRACTICE_SPACE[1] - PRACTICE_SPACE[0])
PRACTICE_SPACE_DIAMETER_NORMED = np.sqrt(PRACTICE_SPACE.shape[1])
PRACTICE_SPACE_MODE = np.linalg.norm(PRACTICE_SPACE[2])
PRACTICE_SPACE_MODE_RATIO = PRACTICE_SPACE_MODE / PRACTICE_SPACE_DIAMETER
PRACTICE_SPACE_RADIUS_RATIO = max(PRACTICE_SPACE_MODE_RATIO, 1 - PRACTICE_SPACE_MODE_RATIO)

IS_RAND_SAMPLING_GOAL = False # learn to generalize in whole (noisy) practice-space
IS_TERMINATION_ON_LEAVING_PRACTICE_SPACE = True # radically decrease state-/searchspace
IS_TRAJECTORY_HALVING = True # further attempts to re-improve current trajectory

# TODO no halving on truncation?

# "breadcrumbing"
# rewards:
#   too frequent => no movement (idleness, fast-narrow conv.)
#   too sparse => no improvement (randomness, slow-broad conv.)
#   too painful => no courage (fearful, no conv.)
IS_GOAL_REWARD_ADAPTIVE_THRESHOLD = True
GOAL_REWARD_THRESHOLD_MIN = 0.0 * PRACTICE_SPACE_RADIUS_RATIO * PRACTICE_SPACE_DIAMETER_NORMED # REWARD TOLERANCE
GOAL_REWARD_THRESHOLD_MAX = 1.0 * PRACTICE_SPACE_RADIUS_RATIO * PRACTICE_SPACE_DIAMETER_NORMED

GOAL_ADAPTIVE_REWARD_THRESHOLD_MEAN = 0.05 # [0,1] REWARD SPARSITY - adapts threshold for specific rewards mean (hold constant difficulty level)
GOAL_ADAPTIVE_REWARD_THRESHOLD_RATE = 0.05 # REWARD ADAPTABILITY - how fast it adapts per step (~how well it holds the rewards mean (=sparsity)) 

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
        obspace = spaces.Box(-np.inf, np.inf, shape=(PRACTICE_SPACE.shape[1],), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        # once
        self.goal_reward_threshold = GOAL_REWARD_THRESHOLD_MIN
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

        achieved_goal = np.array((height, velocity, angle, n_contact_after, angle_thigh, is_moving_forward))
        achieved_goal_norm = self._normalize(achieved_goal, PRACTICE_SPACE[0], PRACTICE_SPACE[1])
        desired_goal_norm = self._normalize(self.desired_goal, PRACTICE_SPACE[0], PRACTICE_SPACE[1])

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

        goaldiff_weighted = PRACTICE_SPACE[3] * np.array([achieved_goal - desired_goal])
        # distance/accuracy (> at-least-only (needs control from both sides))
        goaldistance = np.linalg.norm(goaldiff_weighted, axis=-1)
        if goaldistance.shape[-1] == 1:
            # single step (no replay)
            self.ep_goal_distances.append(goaldistance[0])

        reward = (goaldistance < self.goal_reward_threshold).astype(np.float64)
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

        if IS_GOAL_REWARD_ADAPTIVE_THRESHOLD:
            if self.ep_rewards_mean > GOAL_ADAPTIVE_REWARD_THRESHOLD_MEAN:
                self.goal_reward_threshold -= GOAL_ADAPTIVE_REWARD_THRESHOLD_RATE
            else:
                self.goal_reward_threshold += GOAL_ADAPTIVE_REWARD_THRESHOLD_RATE
            self.goal_reward_threshold = max(GOAL_REWARD_THRESHOLD_MIN, self.goal_reward_threshold)
            self.goal_reward_threshold = min(GOAL_REWARD_THRESHOLD_MAX, self.goal_reward_threshold)

        terminated = False
        truncated = False

        # termination shaping?
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        dims_outside, = np.where(np.logical_or(obs['achieved_goal'] < 0, obs['achieved_goal'] > 1))
        
        if len(dims_outside) > 0:
            print('outside practice space!', PRACTICE_SPACE_LABELS[dims_outside], obs['achieved_goal'][dims_outside], sep=' ')
            terminated = IS_TERMINATION_ON_LEAVING_PRACTICE_SPACE
            reward = 0

        if self.ep_num_steps > 1000:
            print('truncated.')
            truncated = True

        # ---------

        if self.render_mode == "human":
            self.render()

        result = obs, float(reward), terminated, truncated, info
        return result
 

    def reset_model(self):
        obs_init = None

        if self.ep_obs_cur:
            # print(self.ep_goal_distances)
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_num_steps', self.ep_num_steps)
            print('ep_goal_distance_min', min(self.ep_goal_distances))
            print('ep_goal_convergence_per_step', (max(self.ep_goal_distances) - min(self.ep_goal_distances)) / self.ep_num_steps)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('threshold_goaldistance', self.goal_reward_threshold)
            print('\n')

        if self.ep_num_steps > 1:
            ep_goal_distance_min = min(self.ep_goal_distances)

            # if IS_TRAJECTORY_HALVING and (self.ep_rewards_mean > self.last_ep_rewards_mean):
            if IS_TRAJECTORY_HALVING and (ep_goal_distance_min < self.last_ep_goal_distance_min):
                print('halving!')
                # idx_half = len(self.ep_states)//2
                # idx_highest_goalconvergence = np.argmin(np.gradient(self.ep_goal_distances))
                idx_lowest_goaldistance = np.argmin(self.ep_goal_distances)

                idx_halving = idx_lowest_goaldistance
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
        if IS_RAND_SAMPLING_GOAL:
            # https://en.wikipedia.org/wiki/Triangular_distribution
            return np.random.triangular(PRACTICE_SPACE[0], PRACTICE_SPACE[2], PRACTICE_SPACE[1])
        else:
            return PRACTICE_SPACE[2]


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
        # adaptive threshold: per episode vs. per training
        # self.threshold_goaldistance = THRESHOLD_ACCURACY_NORMED_ABS
        print('desired_goal ', self.desired_goal)
