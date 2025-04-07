from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
import numpy as np

# ref https://www.youtube.com/watch?v=irkXnpZP89s
# records
# record 1.0    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aab0ae/le-walker2d-v4/17-58-22/rl_model_finished
#   400k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43/rl_model_finished
#   500k, velo1 /home/t14/Documents/tuhh/dsf/Scilab-RL/data/231edcb/le-walker2d-v4/21-10-43_restored/rl_model_finished

#   400k, velo2, no practicemode    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/48779a8/le-walker2d-v4/00-14-47/rl_model_finished

# forming: goal + termination
# reward-trickling ("breadcrumbing")

# glossar
# GOALSPACE_DESIRED := goalstate -+ threshold
# (GOAL-)STATE_ACHIEVED := current state of step
# PRACTICE-/TRAINSPACE := all reasonable states to act from

# TODO goal: time-dim. vs. infinite (non-episodic), stand-up? 

IS_PRACTICE_MODE = True

IS_RAND_SAMPLING_GOAL = IS_PRACTICE_MODE
# what if contradictory goals?
# practice/train space
# = same as all(!) reasonable/possible/unconstrained states (incl. start-state) (transfer-learning?)
# TODO how to find perfect practice space? (watch/oversee (at the start)! eg. too strict vs. too lax, also learn to (slightly) recover?)
# intervals
#   smaller => faster, focussed
#   larger => slower, more universal
# generally: the more goal dims., the better? ("more experienced coach")
# TODO goal analysis (eg. most failed dim.) on eval
GOAL_SPACE_DESIRED = np.array([
    # height, velocity, angle, contact
    [0.7, -2.0, -1.0, 0], # min
    [2.0, 3.5, 1.0, 3], # max
    [1.1, 2.5, 0.5, 1], # mode
    [1.0, 2.0, 1.0, 1.0] # weight (TODO any impact?)
])

# TODO no halving on truncation
IS_TRAJECTORY_HALVING = IS_PRACTICE_MODE

# TODO as rel. factor?
THRESHOLD_ACCURACY_ABS = 0.3 # REWARD TOLERANCE - wont be less, needs some scaling with dims.?
IS_ADAPTIVE_ACCURACY_THRESHOLD = IS_PRACTICE_MODE
# "breadcrumbing"
ADAPTIVE_ACCURACY_THRESHOLD_TARGET_REWARDS_MEAN = 0.1 # [0,1] REWARD SPARSITY - adapts threshold for specific rewards mean (hold constant difficulty level)
ADAPTIVE_ACCURACY_THRESHOLD_STEP_ABS = 0.1 # REWARD ADAPTABILITY - how fast it adapts (~how well it holds the rewards mean (=sparsity)) 


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
        obspace = spaces.Box(-np.inf, np.inf, shape=(GOAL_SPACE_DESIRED.shape[1],), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=obspace,
                achieved_goal=obspace,
                observation=orig_obspace,
            )
        )

        # once
        self.threshold_accuracy = THRESHOLD_ACCURACY_ABS
        self.desired_goal = None
        self.last_ep_rewards_mean: float = 0
        # every ep
        self._reset_episode()
        print('le-walker-2d initialized.')


    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = np.clip(self.data.qvel.flat.copy(), -10, 10)
        self.ep_states.append((qpos, qvel))

        if self._exclude_current_positions_from_observation:
            qpos = qpos[1:]

        observation = np.concatenate((qpos, qvel)).ravel()
        height, velocity, angle = qpos[0], qvel[0], qpos[1]
        n_contact = self.data.ncon

        achieved_goal = np.array((height, velocity, angle, n_contact))
        achieved_goal_norm, desired_goal_norm = self._normalize(achieved_goal, self.desired_goal, GOAL_SPACE_DESIRED[0], GOAL_SPACE_DESIRED[1])
        # print(achieved_goal)

        obs = dict(
                observation=observation,
                achieved_goal=achieved_goal_norm,
                desired_goal=desired_goal_norm,
            )

        return obs
    

    def _normalize(self, achieved_goal, desired_goal, min_goal, max_goal):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        achieved_goal = (achieved_goal - min_goal) / (max_goal - min_goal)
        desired_goal = (desired_goal - min_goal) / (max_goal - min_goal)
        
        return achieved_goal, desired_goal


    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        goal_diff = np.array([achieved_goal - desired_goal]) * GOAL_SPACE_DESIRED[3]
        # distance/accuracy (~min-max, != logical_and(), > at-least-only (needs control from both sides))
        accuracy = np.linalg.norm(goal_diff, axis=-1)
        reward = (accuracy < self.threshold_accuracy).astype(np.float64)
        return reward


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        info = {}
        obs = self._get_obs()
        self.ep_obs_cur = obs

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        if reward and self.ep_first_reward_step < 0:
            self.ep_first_reward_step = self.ep_num_steps

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if IS_ADAPTIVE_ACCURACY_THRESHOLD:
            if self.ep_rewards_mean > ADAPTIVE_ACCURACY_THRESHOLD_TARGET_REWARDS_MEAN:
                self.threshold_accuracy -= ADAPTIVE_ACCURACY_THRESHOLD_STEP_ABS
            else:
                self.threshold_accuracy += ADAPTIVE_ACCURACY_THRESHOLD_STEP_ABS
            self.threshold_accuracy = max(THRESHOLD_ACCURACY_ABS, self.threshold_accuracy)

        terminated = False
        truncated = False

        # termination shaping?
        # termination if irrevertible?
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        if (obs['achieved_goal'] < 0).any() or (obs['achieved_goal'] > 1).any():
            print('left practice space! ', obs['achieved_goal'])
            terminated = True
            reward = 0

        velocity = obs['observation'][8]
        if self.ep_num_steps > 300 and velocity < 0.3:
            print('not forward!', velocity)
            terminated = True
            reward = 0

        # mean_velocity_all = np.mean(np.abs(obs['observation']))
        # if mean_velocity_all < 0.2:
        #     print('standing still!', mean_velocity_all)
        #     terminated = True
        #     reward = 0

        if self.ep_num_steps > 1000:
            print('truncated.')
            truncated = True

        # ---------

        if self.render_mode == "human":
            self.render()

        result = obs, reward, terminated, truncated, info
        return result
 

    def reset_model(self):
        if self.ep_obs_cur:
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_num_steps', self.ep_num_steps)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('ep_threshold_accuracy', self.threshold_accuracy)
            print('\n')

        if IS_TRAJECTORY_HALVING and (self.ep_rewards_mean > self.last_ep_rewards_mean):
            # TODO add noise?
            print('halving!')
            self.last_ep_rewards_mean = self.ep_rewards_mean
            # state_halfway = self.ep_states[len(self.ep_states)//2]
            qpos, qvel = self.ep_states[len(self.ep_states)//2]
            self.set_state(qpos, qvel)
            ep_obs_init = self._get_obs()

        else:
            # new goal
            self.desired_goal = self._get_goal()
            self.last_ep_rewards_mean = 0
            ep_obs_init = super().reset_model()
            
        self._reset_episode()
        return ep_obs_init


    def _get_goal(self):
        if IS_RAND_SAMPLING_GOAL:
            # https://en.wikipedia.org/wiki/Triangular_distribution
            return np.random.triangular(GOAL_SPACE_DESIRED[0], GOAL_SPACE_DESIRED[2], GOAL_SPACE_DESIRED[1])
        else:
            return GOAL_SPACE_DESIRED[2]


    def _reset_episode(self):
        self.ep_rewards_mean: float = 0
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_obs_cur = None
        self.ep_states = []
        print('desired_goal ', self.desired_goal)
