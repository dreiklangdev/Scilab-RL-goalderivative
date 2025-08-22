
import numpy as np
import time
import random
import copy
import logging
import git
import uuid
from types import SimpleNamespace
from . import hand_imitation_cfg as cfg
from . import autoencoder
from .hand_imitation_subproc_vidcap import VidCapSingletonProc

from gymnasium import spaces
from gymnasium.wrappers.utils import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback

from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

import multiprocessing
import glob
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.rcParams["figure.raise_window"] = False

import cv2
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D

import mujoco
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import logging
LOG = logging.getLogger(__name__)
LOG.propagate = False
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter(fmt='%(message)s'))
LOG.addHandler(consoleHandler)
multiprocessing.log_to_stderr(logging.DEBUG)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PATH_GIT_WORKING_DIR = git.Repo('.', search_parent_directories=True).working_tree_dir

# https://ai.stackexchange.com/questions/20903/what-is-the-difference-between-training-and-testing-in-reinforcement-learning
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
# https://stable-baselines3.readthedocs.io/en/master/modules/her.html
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
# https://huggingface.co/sb3
# https://thegradient.pub/learning-from-humans-what-is-inverse-reinforcement-learning/
# https://github.com/yosider/ml-agents-1/blob/master/docs/Training-SAC.md
# https://old.reddit.com/r/MachineLearning/comments/xfmqny/d_what_happened_to_reinforcement_learning/
# https://paperswithcode.com/
# https://paperswithcode.com/task/humanoid-control
# https://medium.com/biased-algorithms/reward-function-in-reinforcement-learning-c9ee04cabe7d

# https://chuoling.github.io/mediapipe/solutions/pose.html
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
# https://saiwa.ai/blog/openpose-vs-mediapipe/
# https://jetson-docs.com/libraries/mediapipe/overview

# https://pytorch.org/rl/0.6/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
# https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb
# https://github.com/google-deepmind/mujoco/issues/85

# https://github.com/huggingface/lerobot
# https://www.reddit.com/r/reinforcementlearning/comments/vqb2wu/tips_and_tricks_for_rl_from_experimental_data/
# https://gymnasium.farama.org/tutorials/gymnasium_basics/load_quadruped_model/

# https://github.com/google-ai-edge/mediapipe/issues/5325
# https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
# https://github.com/google-ai-edge/mediapipe/issues/5325
# https://github.com/google-deepmind/mujoco/issues/85
# TODO terminate on missing pose detection?

# https://www.reddit.com/r/reinforcementlearning/comments/18v50ai/conventions_to_write_a_custom_vectorized_gym/
# https://github.com/roboterax/humanoid-gym

# vs. scilab-rl
# no envpool, no mjx (3.0)
# https://github.com/sail-sg/envpool
# https://mujoco.readthedocs.io/en/stable/mjx.html
# https://github.com/google-deepmind/mujoco_menagerie
# https://mujoco.readthedocs.io/en/stable/models.html
# https://github.com/clvrai/awesome-rl-envs?tab=readme-ov-file#humanoid
# https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
# https://github.com/atabakd/MuJoCo-Tutorials/blob/master/include/mjdata.h

# https://cookbook.chromadb.dev/running/performance-tips/#__tabbed_1_1

# https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
# https://www.ce.cit.tum.de/mmk/shgd/
# https://github.com/hukenovs/hagrid/blob/master/images/gestures.png
# https://www.qualcomm.com/developer/software/jester-dataset#datasetdetails

# https://scikit-learn.org/stable/model_persistence.html

IS_OBSPACE_PAD_TO_NEXT_BASE_2 = False

# TODO obs appender func with limits warning (for normalization(!))
# TODO persist models (action, pca, zscale?)
# 0.5M pcaObsGoal, gamma0   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8c4bd85/le-hand-imitation-v1/17-03-29_restored/rl_model_finished
# 0.5M bothsidedAllCompass, noRecordRewarding, meanSampling /home/t14/Documents/tuhh/dsf/Scilab-RL/data/3839639/le-hand-imitation-v1/12-56-20/rl_model_finished
# 0.5M bothsidedAllCompass, RecordRewarding, uniSampling /home/t14/Documents/tuhh/dsf/Scilab-RL/data/96a1b45/le-hand-imitation-v1/14-19-54/rl_model_finished
# 0.5M relativeGoalObs, bothsidedAllCompass, noRecordRewarding, uniSampling /home/t14/Documents/tuhh/dsf/Scilab-RL/data/96a1b45/le-hand-imitation-v1/14-19-54/rl_model_finished
# 0.5M pcaGoal0.5, goaldistHistory, relativeGoalObs, bothsidedAllCompass, unisampling /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2dbf116/le-hand-imitation-v1/23-12-40/rl_model_finished
# 1.0M /home/t14/Documents/tuhh/dsf/Scilab-RL/data/2dbf116/le-hand-imitation-v1/23-12-40_restored/rl_model_finished
# 0.5M relativeGoalObsReducedOnly (much better precision, better/closer results) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d820e9e/le-hand-imitation-v1/19-18-05/rl_model_finished
# 0.5M noWorldPosObs (only worldderivs), fullDerivsObs, derivFrontRewarding   restore_policy=/home/t14/Documents/tuhh/dsf/Scilab-RL/data/4f83e3b/le-hand-imitation-v1/15-12-59/rl_model_finished
# 0.3M noWorldObs whatsoever: pca0.5-reduced goalobs only (goaldiffs/-derivs, goaldist/-derivs)    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/7ab0a84/le-hand-imitation-v1/22-11-37/rl_model_finished
# 0.5M(!) noWorldObs whatsoever: pca0.5-reduced goalobs only (goaldiffs/-derivs, goaldist/-derivs), allGestures    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/300e824/le-hand-imitation-v1/23-29-18/rl_model_finished
# 0.5M no posit. rewards, noWorldObs whatsoever: pca0.5-reduced goalobs only (goaldiffs/-derivs, goaldist/-derivs), allGestures   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/244f064/le-hand-imitation-v1/13-05-44/rl_model_finished
# 0.5M goalMom3, no posit. rewards (only deriv.), noWorldObs whatsoever: pca0.5-reduced goalobs only (goaldiffs/-derivs, goaldist/-derivs), allGestures, validSet   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/244f064/le-hand-imitation-v1/22-59-03/rl_model_finished


# # bad goal, less fluent, very sharp, bad generality, more efficient
# 500k, k3, (256,256,256), 0.001,  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ac87bda/le-hand-imitation-v1/16-45-00/rl_model_finished

# # (current best) ok goal, more fluent
# 1M, k4, (256,256,128), 0.001, /home/t14/Documents/tuhh/dsf/Scilab-RL/data/d246874/le-hand-imitation-v1/12-54-38/rl_model_finished

# # worse, no imitation, slightly better start though (looked promising)
# 0.5M, k4, (1024,1024,1024), 0.001,  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ac87bda/le-hand-imitation-v1/18-54-15/rl_model_finished


class HandImitationEnv(HumanoidEnv):


    def __init__(self, is_eval=False, is_render=True, is_plot=True, submodels=None, log_level=logging.INFO):
        LOG.setLevel(log_level)

        HumanoidEnv.__init__(self,
                             exclude_current_positions_from_observation=True,
                             width=cfg.General.RENDER_IMAGE_SIZE,
                             height=cfg.General.RENDER_IMAGE_SIZE,
                             xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_hand/adroit_hand/adroit_relocate.xml')
        # BaseCallback(HandImitationEnv, self).__init__(verbose=0)
        self.frame_skip: 5 = cfg.General.FRAMESKIP_STEP

        assert cfg.General.STEPSKIP_DETECT >= cfg.General.STEPSKIP_DETECT and cfg.General.STEPSKIP_DETECT >= cfg.General.STEPSKIP_DETECT, 'cannot plot in a step with no pose render (and detection'

        self.cfg = cfg
        self.is_render = is_render
        self.tr_is_eval = is_eval
        self.is_plot = is_plot
        self.outfile_tr_multigoal_lastmeans = open('tr_multigoal_lastmeans.dat', 'a')
        self.outfile_tr_goalprogresses = open('goalprogresses.dat', 'a')

        # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
        self.landmarker_options_achieved = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=PATH_GIT_WORKING_DIR + '/mediapipe/model/hand_landmarker.task',
                # cpu vs gpu
                # https://forums.developer.nvidia.com/t/how-to-install-opengl-libs-of-nvidia/175409
                # https://stackoverflow.com/questions/77707532/how-to-check-for-and-enforce-gpu-usage-for-mediapipe-frame-processing/79202595#79202595
                # https://stackoverflow.com/questions/74048393/can-i-speed-up-processing-live-video-from-webcam
                # https://raw.githubusercontent.com/opencv/opencv_zoo/main/benchmark/color_table.svg
                # prime-select nvidia
                # glxinfo | grep -i opengl
                # MUJOCO_GL=egl|glfw|osmesa %python ...% (glfw seems fastest)
                delegate=BaseOptions.Delegate.GPU),
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1)
        self.landmarker_achieved = HandLandmarker.create_from_options(self.landmarker_options_achieved)

        self.landmarker_options_desired = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=PATH_GIT_WORKING_DIR + '/mediapipe/model/hand_landmarker.task',          
                delegate=BaseOptions.Delegate.GPU),
            running_mode=VisionRunningMode.IMAGE,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1)
        self.landmarker_desired = HandLandmarker.create_from_options(self.landmarker_options_desired)

        obspace_total_dims = 0

        # world obs
        # obspace_total_dims += self.observation_space.shape[0] # super
        # obspace_total_dims += self.data.qpos.flatten().shape[0] # super-qpos
        # no contact forces available https://github.com/openai/gym/issues/1541
        # obspace_total_dims += self.data.cfrc_ext.flatten().shape[0] # super-actuatorforce
        # obspace_total_dims += self.data.qpos.flatten().shape[0] * np.sum(range(self.cfg.General.WORLD_DERIV_ORDERS + 1)) # superpos-derivs
        # obspace_total_dims += self.data.qpos.flatten().shape[0] * self.cfg.General.OBS_WORLD_DERIV_ORDERS # superpos-derivs_front

        # obspace_total_dims += self.data.qpos.flatten().shape[0] * self.cfg.General.OBS_WORLD_DERIV_ORDERS # last superpos-derivs

        # achieved obs
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # achieved: pose
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # achieved: pose_reduced

        # desired obs
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # desired: pose
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # desired: pose_reduced
        
        # goal obs
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # goaldiff
        obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # goaldiff_reduced
        obspace_total_dims += 2 * cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # goaldiff_recent
        obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # goaldiffs
        obspace_total_dims += 1 # goaldist
        obspace_total_dims += self.cfg.General.GOAL_DERIV_ORDERS + 1 # goaldists_recent
        obspace_total_dims += np.sum(range(self.cfg.General.GOAL_DERIV_ORDERS + 1)) # goalderivs


        # obspace_total_dims += 2 ** self.cfg.General.GOAL_DERIV_ORDERS # goalconvs_recent
        # obspace_total_dims += 1 # goalangle

        # meta obs
        if self.cfg.MetaObservation.IS_ENABLED:
            obspace_total_dims += 1 # steps_diverging_left https://arxiv.org/abs/1712.00378
            obspace_total_dims += 1 # goal-hash

        # reward obs
        obspace_total_dims += 1 + self.cfg.General.OBS_REWARD_HISTORY_LENGTH
        obspace_total_dims += self.cfg.General.REWARD_DERIV_ORDERS

        # pad to 2^x
        if IS_OBSPACE_PAD_TO_NEXT_BASE_2:
            # https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
            obspace_total_dims = 1 << (obspace_total_dims-1).bit_length()

        observation_space = spaces.Box(-np.inf, np.inf, shape=(obspace_total_dims,), dtype='float64')
        goal_space = spaces.Box(-np.inf, np.inf, shape=(1 + self.cfg.General.GOAL_DERIV_ORDERS,), dtype='float64') # goaldist, goalconv
        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=goal_space,
                achieved_goal=goal_space,
            )
        )


        LOG.debug('observation_space %s', observation_space)
        LOG.info('action_space %s', self.action_space)
        LOG.debug('goal_space %s', goal_space)

        # once        
        self.tr_multigoal_paths = glob.glob(PATH_GIT_WORKING_DIR + '/mediapipe/poses/hand/*.jpg')
        if self.tr_is_eval: 
            self.tr_multigoal_paths = ['cam'] + self.tr_multigoal_paths
        self.tr_multigoal_distrecords = [99999.0] * len(self.tr_multigoal_paths)
        self.tr_multigoal_lastmeans = [99999.0] * len(self.tr_multigoal_paths)
        self.fep_goalid = -1
        self.fep_is_dense = True
        self.tr_learning_started = False

        
        self.buffer_obs_achieved = []
        self.buffer_obs_world = []

        # submodels
        self.zs_scaler_goal = submodels["zs_scaler_goal"]
        self.zs_scaler_obs = submodels["zs_scaler_obs"]
        self.pca_reducer_goal = submodels["pca_reducer_goal"]

        self.pca_goal_modelref = []

        self.ac_model_encobs = autoencoder.Autoencoder(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
        self.recon_loss = nn.MSELoss()
        self.ac_optimizer = optim.Adam(self.ac_model_encobs.parameters(), lr=1e-3)

        self.init_qpos[6] = -1.4 # face towards camera
        self.pose_scale_ratio = 1
        self.tr_n_feps: int = 0
        self.tr_goaldist_min: float = 1
        self.tr_goaldist_max: float = 0
        self.tr_goaldist_mins_mean: float = 0
        self.tr_goaldist_maxs_mean: float = 0
        self.tr_goalconv_min: float = 1
        self.tr_goalconv_max: float = 0
        self.tr_reward_min: float = 1
        self.tr_reward_max: float = 0
        self.tr_num_steps: int = 0
        self.tr_ep_num_steps_max: int = 0
        self.fep_savepoint_steps = 0
        self.fep_savepoint_steps_goal_zone: int = 0
        self.fep_goaldist_init = np.inf
        self.fep_goaldist_min: float = np.inf
        self.fep_goaldist_max: float = 0
        self.fep_goaldims_primary = []
        self.fep_goaldims_secondary = []
        self.fep_rewards_sum = -1
        self.fep_obs_init = None
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goaldist_min: float = np.inf
        self.ep_num_steps: int = 0
        self.fep_goalhash: float = -1
        self.ep_goaldist_min: float = np.inf
        self.ep_goaldist_max: float = 0
        self.ep_goalweight = []
        self.fep_lives = cfg.TrajectoryHalving.MAX_LIVES
        self.ep_rand_videostart = 0
        self.lp_num_steps = 0

        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self.landmarker_achieved = None
        self.landmarker_desired = None
        self.last_ob_pose_achieved = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)
        self.last_ob_desired_pose = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)

        if self.tr_is_eval:
            VidCapSingletonProc.autochange_ev.set()

        if self.is_plot:
            self.parallel_plot_queue = multiprocessing.Queue()
            multiprocessing.Process(target=parallel_plot, args=((self.parallel_plot_queue,)), daemon=True).start()

        self._reset()
        LOG.debug('le-walker-2d initialized.')


    def step_passive(self, action):
        info = {}
        info['success'] = False

        # reduce action space?
        # action = np.clip(action, -np.pi/2, np.pi/2)
        self.do_simulation(action, self.frame_skip)
        self.ep_num_steps += 1
        self.tr_num_steps += 1

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.ep_states.append((qpos, qvel))
        self.ep_actions.append(action)

        obs = self._get_obs()
        goaldist = obs['achieved_goal'][0]
        goalconv = obs['achieved_goal'][1]
        self.ep_goaldists.append(goaldist)
        self.ep_goalconvs.append(goalconv)
        self.ep_dictobs.append(obs)
        self.ep_current_obs = obs


        if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
            self.ep_reward_threshold = np.random.normal(0, np.mean(self.ep_goaldists))


        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info).item()
        self.ep_rewards.append(reward)


        # records
        if goaldist < self.ep_reward_threshold:
            self.ep_num_steps_goal_zone += 1

        if goaldist < self.tr_goaldist_min:
            self.tr_goaldist_min = goaldist

        if goaldist > self.tr_goaldist_max:
            self.tr_goaldist_max = goaldist        

        if goaldist < self.ep_goaldist_min:
            self.ep_goaldist_min = goaldist
           
        if goaldist > self.ep_goaldist_max:
            self.ep_goaldist_max = goaldist

        if goaldist < self.fep_goaldist_min:
            self.fep_goaldist_min = goaldist
            self.fep_lives = cfg.TrajectoryHalving.MAX_LIVES
            self.ep_last_step_approached = self.ep_num_steps

            # if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
            #     self.ep_reward_threshold = goaldist
        
        if goaldist > self.fep_goaldist_max:
            self.fep_goaldist_max = goaldist

        if goalconv < self.tr_goalconv_min:
            self.tr_goalconv_min = goalconv

        if goalconv > self.tr_goalconv_max:
            self.tr_goalconv_max = goalconv

        if goalconv > 0:
            self.ep_num_steps_conv += 1

        if reward < self.tr_reward_min:
            self.tr_reward_min = reward

        if reward > self.tr_reward_max:
            self.tr_reward_max = reward

        if self.ep_num_steps > self.tr_ep_num_steps_max:
            self.tr_ep_num_steps_max = self.ep_num_steps


        terminated = False
        truncated = False

        if self.tr_learning_started:
            # if goaldist - self.tr_multigoal_distrecords[self.fep_goalid] <= 0.0:
            #     LOG.debug('goal distrecord reached.')
            #     self.tr_multigoal_distrecords[self.fep_goalid] = goaldist
            #     # may hinder compass (follow) learning? at least hinders initial exploration?
            #     reward = 1
            if goaldist <= self.tr_multigoal_distrecords[self.fep_goalid]:
                # LOG.debug('goal distrecord reached or improved %s', goaldist)
                self.tr_multigoal_distrecords[self.fep_goalid] = goaldist
                # reward = 1

        # space constraint
        # reckless training (no penalties, fast respawn)
        if not self.tr_is_eval and self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE and self.ep_num_steps > self.cfg.PracticeSpace.STEPS_INVINCIBLE_SPAWN:

            # # TODO only in goal-hold phase? (goal-reach may need divergent steps...)
            # if len(self.ep_goalconvs) > MAX_DIVERGENT_STEPS and not np.argmax(np.array(self.ep_goalconvs[-MAX_DIVERGENT_STEPS:]) > 0):
            # if (self.ep_num_steps - self.ep_num_steps_conv) > self.cfg.PracticeSpace.MAX_DIVERGENT_STEPS:
            #     terminated = True
            #     # reward = -1
            #     LOG.info('TOO MANY DIVERGENT STEPS.')

            if (self.ep_num_steps - self.ep_last_step_approached) > 1000:
                terminated = True
                # reward = -1
                LOG.info('NO APPROACHING STEPS. %s', 1000)

            # min. convergence terminate? ("flaming wall")
            
            # elif self.ep_count_fails_pose_detection > 10:
            #     LOG.info('TOO MANY DETECTION FAILURES. (better detection at higher res.?)')
            #     terminated = True
            #     self.ep_lives -= 1
            #     reward = 0

        if self.tr_is_eval:
            if self.ep_num_steps >= 1000:
                truncated = True



        self.ep_goaldist_mean = (((self.ep_num_steps - 1) * self.ep_goaldist_mean) + goaldist) / (self.ep_num_steps)
        self.tr_multigoal_lastmeans[self.fep_goalid] = self.ep_goaldist_mean
        self.ep_rewards_mean = (((self.ep_num_steps - 1) * self.ep_rewards_mean) + reward) / (self.ep_num_steps)


        # also skip first buggy render
        if self.tr_n_feps == 1 or self.ep_num_steps > self.cfg.PracticeSpace.MAX_STEPS_EPISODE_TRUNCATION:
            LOG.info('TRUNCATED.')
            truncated = True
            is_success = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            info['success'] = is_success

        # goalprogress
        if self.ep_goaldists and self.ep_goaldists[0] > 0:
            self.ep_goalprogress = (self.ep_goaldists[0] - self.ep_goaldists[-1]) / self.ep_goaldists[0]
            self.ep_goalprogress = max(0, self.ep_goalprogress)
            info['goalprogress'] = self.ep_goalprogress

        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(np.round(reward, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goaldist', str(np.round(goaldist, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goalprogress', str(np.round(self.ep_goalprogress, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goalseek', str(goaldist > self.ep_reward_threshold))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'meandist_r', str(np.round(goaldist / self.ep_goaldist_mean, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'ep_rewards_mean', str(np.round(self.ep_rewards_mean, 2)))
            # ep_goalzone_per_step = np.round(self.ep_num_steps_goal_zone /  self.ep_num_steps, 2)
            # human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'ep_goalzone_per_step', str(ep_goalzone_per_step))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'fep_goalid', str(self.fep_goalid))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'fep_is_dense', str(self.fep_is_dense))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'fep_rewards_sum', str(np.round(self.fep_rewards_sum, 2)))
            human_viewer.render()

        self.ep_current_reward = reward
        self.ep_rewards_sum += reward
        self.fep_rewards_sum += reward

        result = obs, reward, terminated, truncated, info
        return result


    def step(self, action):
        num_steps_passive = 0
        num_steps_passive_max = 1 # adaptive? (eg. dont scout anymore at goal(-keeping))
        total_reward = 0
        last_reward = None
        term = False
        trunc = False
        # retro-future rewarding ("drone/scout send+collect")
        # TODO add action noise?
        # neg. passive steps should also been cumulated (and learned)
        # while num_steps_passive < num_steps_passive_max and not (term or trunc):
        while num_steps_passive < num_steps_passive_max and (not term or not trunc):
            obs, reward, term, trunc, info = self.step_passive(action)
            num_steps_passive += 1
            total_reward += reward

            if last_reward:
                if reward != last_reward:
                    # change: needs control
                    break
            last_reward = reward

        # TODO pos. passive steps are not (actively) learned/seen by NN (yet)? (only passively by retro/future is enough? maybe no need to active step in?)
        return obs, total_reward, term, trunc, info


    # obs = achieved_obs + metaobs
    # TODO keep obs keys/indices mapping (eg. dict, vs. "counting")
    def _get_obs(self):
        obs = np.array([])


        # =========== WORLD OBS
        world = np.array([])

        # world = np.append(world, super()._get_obs()) # already includes first order (mujoco-computed, possibly different)
        world = np.append(world, self.data.qpos.flatten())
        # world = np.append(world, self.data.cfrc_ext.flatten())

        # obs = np.append(obs, world)

        worldderivs = np.array([])
        worldderiv_orders = self.cfg.General.WORLD_DERIV_ORDERS
        if worldderiv_orders > 0:
            jointpos_recent = np.array([q[0] for q in self.ep_states[-(worldderiv_orders + 1):]]) # only enough recent posis for all orders
            jointpos_recent = np.pad(jointpos_recent, ((max(0, worldderiv_orders + 1 - jointpos_recent.shape[0]),0),(0,0))) # pad for always enough recent posis

            for i in range(1, worldderiv_orders + 1):
                # worldderivs = np.append(worldderivs, np.diff(jointpos_recent, n=i, axis=0)[-1])
                worldderivs = np.append(worldderivs, np.diff(jointpos_recent, n=i, axis=0))

        # more important than expected/supposed?
        # obs = np.append(obs, worldderivs)

        # obs = np.append(obs, self.ep_last_obs_worldderivs)
        self.ep_last_obs_worldderivs = worldderivs


        # ========= ACHIEVED OBS (proprioception)
        obs_achieved = np.array([])

        is_eval_vidcap = self.tr_is_eval and self.fep_goalid == 0
        stepskip_detect = 8 if is_eval_vidcap else cfg.General.STEPSKIP_DETECT
        if self.ep_num_steps == 1 or self.ep_num_steps % stepskip_detect == 0:
            if is_eval_vidcap:
                self.desired_imgdata = VidCapSingletonProc.parallel_vidcap_queue.get()
            self.desired_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.desired_imgdata.copy())
            self.desired_pose = self.landmarker_desired.detect(self.desired_img)

        desired_img = self.desired_img
        desired_pose = self.desired_pose
        desired_pose_orig = copy.deepcopy(desired_pose)
        achieved_pose = copy.deepcopy(desired_pose)

        if desired_pose.hand_landmarks:
            # if desired_pose.hand_landmarks[0][0].x != 0.0:
            # TODO redo once if new desired pose?
            # center desired origin?
            translation_x = desired_pose.hand_landmarks[0][0].x
            translation_y = desired_pose.hand_landmarks[0][0].y
            translation_z = desired_pose.hand_landmarks[0][0].z
            for landmark in desired_pose.hand_landmarks[0]:
                landmark.x -= translation_x
                landmark.y -= translation_y
                landmark.z -= translation_z

            norm_v = np.linalg.norm([
                desired_pose.hand_landmarks[0][0].x - desired_pose.hand_landmarks[0][9].x
                + desired_pose.hand_landmarks[0][9].x - desired_pose.hand_landmarks[0][10].x
                + desired_pose.hand_landmarks[0][10].x - desired_pose.hand_landmarks[0][11].x
                + desired_pose.hand_landmarks[0][11].x - desired_pose.hand_landmarks[0][12].x,
                
                desired_pose.hand_landmarks[0][0].y - desired_pose.hand_landmarks[0][9].y
                + desired_pose.hand_landmarks[0][9].y - desired_pose.hand_landmarks[0][10].y
                + desired_pose.hand_landmarks[0][10].y - desired_pose.hand_landmarks[0][11].y
                + desired_pose.hand_landmarks[0][11].y - desired_pose.hand_landmarks[0][12].y,

                desired_pose.hand_landmarks[0][0].z - desired_pose.hand_landmarks[0][9].z
                + desired_pose.hand_landmarks[0][9].z - desired_pose.hand_landmarks[0][10].z
                + desired_pose.hand_landmarks[0][10].z - desired_pose.hand_landmarks[0][11].z
                + desired_pose.hand_landmarks[0][11].z - desired_pose.hand_landmarks[0][12].z,
                ])

            norm_u = np.linalg.norm([
                self.data.geom('V_wrist').xpos[0] - self.data.geom('V_mfknuckle').xpos[0]
                + self.data.geom('V_mfknuckle').xpos[0] - self.data.geom('V_mfproximal').xpos[0]
                + self.data.geom('V_mfproximal').xpos[0] - self.data.geom('V_mfmiddle').xpos[0]
                + self.data.geom('V_mfmiddle').xpos[0] - self.data.geom('V_mfdistal').xpos[0],

                self.data.geom('V_wrist').xpos[1] - self.data.geom('V_mfknuckle').xpos[1]
                + self.data.geom('V_mfknuckle').xpos[1] - self.data.geom('V_mfproximal').xpos[1]
                + self.data.geom('V_mfproximal').xpos[1] - self.data.geom('V_mfmiddle').xpos[1]
                + self.data.geom('V_mfmiddle').xpos[1] - self.data.geom('V_mfdistal').xpos[1],
                
                self.data.geom('V_wrist').xpos[2] - self.data.geom('V_mfknuckle').xpos[2]
                + self.data.geom('V_mfknuckle').xpos[2] - self.data.geom('V_mfproximal').xpos[2]
                + self.data.geom('V_mfproximal').xpos[2] - self.data.geom('V_mfmiddle').xpos[2]
                + self.data.geom('V_mfmiddle').xpos[2] - self.data.geom('V_mfdistal').xpos[2],
                ])

            for i, body_id in enumerate(cfg.General.MJBODY_TO_MPPOSE):
                if body_id:
                    # achieved_pose.hand_landmarks[0][i].x = self.data.body(body_id).xpos[0]
                    # achieved_pose.hand_landmarks[0][i].y = -self.data.body(body_id).xpos[2]
                    # achieved_pose.hand_landmarks[0][i].z = self.data.body(body_id).xpos[1]
                    achieved_pose.hand_landmarks[0][i].x = self.data.geom(body_id).xpos[0]
                    achieved_pose.hand_landmarks[0][i].y = -self.data.geom(body_id).xpos[2]
                    achieved_pose.hand_landmarks[0][i].z = self.data.geom(body_id).xpos[1]

                    # origin: wrist
                    # achieved_pose.hand_landmarks[0][i].x -= self.data.body('wrist').xpos[0]
                    # achieved_pose.hand_landmarks[0][i].y -= self.data.body('wrist').xpos[2]
                    # achieved_pose.hand_landmarks[0][i].z -= self.data.body('wrist').xpos[1]

                    achieved_pose.hand_landmarks[0][i].x *= 1 / norm_u
                    achieved_pose.hand_landmarks[0][i].y *= 1 / norm_u
                    achieved_pose.hand_landmarks[0][i].z *= 1 / norm_u

                    desired_pose.hand_landmarks[0][i].x *= 1 / norm_v
                    desired_pose.hand_landmarks[0][i].y *= 1 / norm_v
                    desired_pose.hand_landmarks[0][i].z *= 1 / norm_v

                    # translate: desired-wrist
                    # achieved_pose.hand_landmarks[0][i].x += desired_pose.hand_landmarks[0][0].x
                    # achieved_pose.hand_landmarks[0][i].y += desired_pose.hand_landmarks[0][0].y
                    # achieved_pose.hand_landmarks[0][i].z += desired_pose.hand_landmarks[0][0].z

                else:
                    achieved_pose.hand_landmarks[0][i].x = -1
                    achieved_pose.hand_landmarks[0][i].y = -1
                    achieved_pose.hand_landmarks[0][i].z = -1


        ob_achieved_pose = np.array([])
        if achieved_pose.hand_landmarks:
            # only first detected pose
            ob_achieved_pose = [(landmark.x, landmark.y, landmark.z) for landmark in achieved_pose.hand_landmarks[0]]            
            ob_achieved_pose = np.array(ob_achieved_pose)[cfg.General.IDS_LANDMARKS_FILTERED]
            ob_achieved_pose = self._normalize_unit_limit(ob_achieved_pose, -2, 2)

        obs_achieved = np.append(obs_achieved, ob_achieved_pose)
        

        # ========= DESIRED OBS
        obs_desired = np.array([])

        ob_desired_pose = self.last_ob_desired_pose
        if desired_pose.hand_landmarks:
            # only first detected pose
            ob_desired_pose = [(landmark.x, landmark.y, landmark.z) for landmark in desired_pose.hand_landmarks[0]]
            ob_desired_pose = np.array(ob_desired_pose)[cfg.General.IDS_LANDMARKS_FILTERED]
            ob_desired_pose = self._normalize_unit_limit(ob_desired_pose, -2, 2)
            self.last_ob_desired_pose = ob_desired_pose

        obs_desired = np.append(obs_desired, ob_desired_pose)


        # ========= GOAL (MODEL)
        goalhash = vector_to_uniform_scalar(ob_desired_pose.flatten(), base=ob_desired_pose.size)
        self.fep_goalhash = goalhash
        goaldiff = np.array([])
        goaldiffs_recent = np.array([])
        goaldiffderivs = np.array([])
        goaldist = self.ep_reward_threshold + 1
        goalderivs = np.array([])
        goalderivs_front = np.array([])
        goaldists_recent = np.array([])

        if desired_pose.hand_landmarks:

            IS_NORMALIZE_Z_SCORE_GOAL = True
            if IS_NORMALIZE_Z_SCORE_GOAL:
                if not self.tr_is_eval:
                    self.zs_scaler_goal.partial_fit(obs_achieved.reshape(1, -1))
                obs_achieved = self.zs_scaler_goal.transform(obs_achieved.reshape(1, -1))[0]
                obs_desired = self.zs_scaler_goal.transform(obs_desired.reshape(1, -1))[0]

            if not self.pca_goal_modelref:
                self.pca_goal_modelref = [np.inf, obs_achieved, obs_achieved]

            SIZE_BUFFER_OBS_ACHIEVED = 1000 # may equal 'algo.learning_starts'
            if len(self.buffer_obs_achieved) <= SIZE_BUFFER_OBS_ACHIEVED:
                self.buffer_obs_achieved.append(obs_achieved)

            IS_PCA_REDUCE_GOAL = True # decorrelation
            PCA_REDUCTION_WEIGHT = 0.5
            if IS_PCA_REDUCE_GOAL:

                if len(self.buffer_obs_achieved) == SIZE_BUFFER_OBS_ACHIEVED and not self.tr_is_eval:
                    self.pca_reducer_goal.partial_fit(self.buffer_obs_achieved)
                    obs_achieved_reduced = self.pca_reducer_goal.transform(self.pca_goal_modelref[1].reshape(1, -1)) @ self.pca_reducer_goal.components_ + self.pca_reducer_goal.mean_ # zca
                    modelconv = np.linalg.norm(self.pca_goal_modelref[2] - obs_achieved_reduced)
                    LOG.debug('goal dims: pca model fitted. %s', modelconv)
                    self.pca_goal_modelref[0] = modelconv
                    self.pca_goal_modelref[2] = obs_achieved_reduced

                if hasattr(self.pca_reducer_goal, 'n_samples_seen_') and self.pca_reducer_goal.n_samples_seen_ > 0:
                    obs_achieved_reduced = self.pca_reducer_goal.transform(obs_achieved.reshape(1, -1)) @ self.pca_reducer_goal.components_ + self.pca_reducer_goal.mean_
                    obs_achieved = (1-PCA_REDUCTION_WEIGHT) * obs_achieved + PCA_REDUCTION_WEIGHT * obs_achieved_reduced[0]

                    obs_desired_reduced = self.pca_reducer_goal.transform(obs_desired.reshape(1, -1)) @ self.pca_reducer_goal.components_ + self.pca_reducer_goal.mean_
                    obs_desired = (1-PCA_REDUCTION_WEIGHT) * obs_desired + PCA_REDUCTION_WEIGHT * obs_desired_reduced[0]

            IS_GOAL_AUTOENCODE = False
            if IS_GOAL_AUTOENCODE:
                if len(self.buffer_obs_achieved) == SIZE_BUFFER_OBS_ACHIEVED:
                    # batch = random.sample(self.ac_buffer_obs_achieved, 1000)
                    batch = [self.buffer_obs_achieved, [obs_desired] * SIZE_BUFFER_OBS_ACHIEVED]

                    # Train
                    tensor = torch.tensor(batch, dtype=torch.float32)
                    tensor_batches = tensor.split(64)  # mini-batch training
                    for epoch in range(20):
                        for batch in tensor_batches:
                            tensor_recon, z = self.ac_model_encobs(batch)
                            recon_loss = self.recon_loss(tensor_recon, batch)
                            decor_loss = decorrelation_loss(z)
                            loss = recon_loss + 0.1 * decor_loss
                            self.ac_optimizer.zero_grad()
                            loss.backward()
                            self.ac_optimizer.step()

                    obs_achieved_tensor = torch.tensor(self.pca_goal_modelref[0], dtype=torch.float32).unsqueeze(0)
                    encobs_achieved = self.ac_model_encobs.encoder(obs_achieved_tensor).detach().numpy().squeeze()
                    encobs_achieved = np.resize(encobs_achieved, obs_achieved.shape)
                    LOG.info('goal dims: autoencode model fitted. %s', np.linalg.norm(self.pca_goal_modelref[2] - encobs_achieved))
                    self.pca_goal_modelref[2] = encobs_achieved

                if self.ac_model_encobs:
                    obs_achieved_tensor = torch.tensor(obs_achieved, dtype=torch.float32).unsqueeze(0)
                    encobs_achieved = self.ac_model_encobs.encoder(obs_achieved_tensor).detach().numpy().squeeze()

                    obs_desired_tensor = torch.tensor(obs_desired, dtype=torch.float32).unsqueeze(0)
                    encobs_desired = self.ac_model_encobs.encoder(obs_desired_tensor).detach().numpy().squeeze()

                    obs_achieved = np.resize(encobs_achieved, obs_achieved.shape)
                    obs_desired = np.resize(encobs_desired, obs_desired.shape)

            if len(self.buffer_obs_achieved) >= SIZE_BUFFER_OBS_ACHIEVED:
                self.buffer_obs_achieved.clear()
                self.tr_learning_started = True

            # combing? (stepwise-combing not working with goalconv-rewards(prev. step goal differs))
            # TODO full randomize weighting? (ie. random generalizing)
            self.ep_goalweight = np.full(obs_desired.shape, 1.0)
            # self.ep_goalweight = np.random.rand(obs_desired.shape[-1])
            # self.ep_goalweight[0] = 1 # base primary dim
            # self.ep_goalweight[1] = 1 # base primary dim
            # goaldims_primary = np.random.randint(2, size=1) # multiple?
            # self.ep_goalweight[goaldims_primary] = 1
            # self.ep_goalweight[goaldims_secondary] = 0.5 # never abandon primary goal in favor of secondary goals
            goaldiff = self.ep_goalweight * (obs_achieved - obs_desired)
            goaldiffs_recent = np.vstack((self.ep_last_goaldiff, goaldiff))
            goaldiffderivs = np.diff(goaldiffs_recent, axis=0)

            # qualitative bottleneck? (0d)
            goaldist = np.linalg.norm(goaldiff, axis=-1)            
            goalderiv_orders = self.cfg.General.GOAL_DERIV_ORDERS
            if goalderiv_orders > 0:
                goaldists_recent = np.array(self.ep_goaldists)
                goaldists_recent = np.append(goaldists_recent, goaldist) # most recent
                goaldists_recent = np.array(goaldists_recent[-(goalderiv_orders + 1):]) # only enough recent goaldists for all orders
                goaldists_recent = np.pad(goaldists_recent, (max(0, goalderiv_orders + 1 - len(goaldists_recent)),0)) # fill up with starting 0s if not enough

                for i in range(1, goalderiv_orders + 1):
                    goalderivs = np.append(goalderivs, np.diff(goaldists_recent, n=i, axis=0))
                    goalderivs_front = np.append(goalderivs_front, goalderivs[-1])

            # goalangle = normalized_angle(self.ep_last_goaldiff, goaldiff) * np.sign(goalderivs[0])
            self.ep_last_goaldiff = goaldiff

        if not desired_pose.hand_landmarks:
            ob_desired_pose = ob_achieved_pose = np.zeros(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
            obs_desired = obs_achieved = np.zeros(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
            goaldiff = np.zeros(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
            goaldiffs_recent = np.zeros(2 * cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
            goaldiffderivs = np.zeros(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
            goaldist = self.ep_reward_threshold + 1
            goaldists_recent = np.zeros(self.cfg.General.GOAL_DERIV_ORDERS + 1)
            goalderivs = np.zeros(np.sum(range(self.cfg.General.GOAL_DERIV_ORDERS + 1)))
            goalderivs_front = np.zeros(self.cfg.General.GOAL_DERIV_ORDERS)


        # obs = np.append(obs, ob_achieved_pose)
        # obs = np.append(obs, ob_desired_pose) # goal
        # obs = np.append(obs, ob_achieved_pose - ob_desired_pose) # goaldimsdiff

        # TODO add  obs_achieved?
        # obs = np.append(obs, obs_achieved)
        # obs = np.append(obs, obs_desired) # goal
        obs = np.append(obs, goaldiff) # goaldimsdiff_reduced
        obs = np.append(obs, goaldiffs_recent) # goaldimsdiff_reduced
        obs = np.append(obs, goaldiffderivs) # goaldimsdiff_reduced

        obs = np.append(obs, goaldist)
        obs = np.append(obs, goaldists_recent)
        obs = np.append(obs, goalderivs)

        # obs = np.append(obs, goalconvs_recent)
        # obs = np.append(obs, goalangle)


        # ========= META OBS

        if self.cfg.MetaObservation.IS_ENABLED:
            obs_meta = np.array([])
            # goalweight_hash = vector_to_uniform_scalar(self.ep_goalweight, len(self.ep_goalweight))
            # obs_meta = np.append(obs_meta, goalweight_hash)
            # obs_meta = np.append(obs_meta, obs_desired.ravel())
            # record_dist = max(0, goaldist - self.tr_goaldist_min)
            # metaobs.append(record_dist) # may hinder retraining of restored policy (record-reset)
            # obs_meta = np.append(obs_meta, np.float_(is_seeking_goal))
            # obs_meta = np.append(obs_meta, is_converging)
            steps_diverged = self.ep_num_steps - self.ep_num_steps_conv
            steps_diverging_left = self.cfg.PracticeSpace.MAX_DIVERGENT_STEPS - steps_diverged
            obs_meta = np.append(obs_meta, steps_diverging_left)
            obs_meta = np.append(obs_meta, goalhash) # goal-hash
            obs = np.append(obs, obs_meta)


        # ========= REWARD OBS

        reward = self.compute_reward(np.array([goaldist] + goalderivs.tolist()), None, None)
        history_length = self.cfg.General.OBS_REWARD_HISTORY_LENGTH
        reward_history = np.resize(self.ep_rewards[-history_length:], history_length)

        rewardderiv_orders = self.cfg.General.REWARD_DERIV_ORDERS
        rewardderivs = np.array([])
        if rewardderiv_orders > 0:
            rewards = np.array(self.ep_rewards)
            rewards = np.append(rewards, reward) # most recent
            rewards = np.array(rewards[-(2 ** rewardderiv_orders):]) # only enough recent goaldists for all orders (2^k)
            rewards = np.pad(rewards, (2 ** rewardderiv_orders,0)) # pad for more than enough recents

            for i in range(1, rewardderiv_orders + 1):
                rewardderivs = np.append(rewardderivs, np.diff(rewards, n=i, axis=0)[-1])

        obs = np.append(obs, reward)
        obs = np.append(obs, reward_history)
        obs = np.append(obs, rewardderivs)


        # may lead to faster and more general training (and prevent overfitting ("always challenging" vs. "too comfortable/stale" training)
        # obs = self._add_noise(obs, -0.1, 0.1)
        # obs = obs + np.random.normal(0, 0.1, size=obs.shape)

        IS_NORMALIZE_Z_SCORE_OBS = True
        if IS_NORMALIZE_Z_SCORE_OBS:
            self.zs_scaler_obs.partial_fit(obs.reshape(1, -1))
            obs = self.zs_scaler_obs.transform(obs.reshape(1, -1))[0]

        if IS_OBSPACE_PAD_TO_NEXT_BASE_2:
            obs = np.pad(obs, (0, self.observation_space['observation'].shape[-1] - obs.shape[-1]))

        achieved_goal = np.array([goaldist] + goalderivs_front.tolist())
        desired_goal = np.zeros(achieved_goal.shape)  # ignored
        dictobs = dict(
            observation=obs,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )

        if self.is_plot and self.ep_num_steps % stepskip_detect == 0:
#            achieved_img_annotated = draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
            desired_img_annotated = draw_landmarks_on_image(desired_img.numpy_view(), desired_pose_orig)
            if self.parallel_plot_queue.empty():
                self.parallel_plot_queue.put_nowait((None, desired_img_annotated, achieved_pose, desired_pose))

        return dictobs


    # class Record:
    #     def __init__(self):
    #         record_min = np.inf
    #         record_max = -np.inf

    #     def get_min(self):
    #         return self.record_min


    # is also used by HER (multi-dim. args.)
    # TODO desired goal may be current dist record
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        if achieved_goal.ndim > 1:
            # LOG.debug('hindsight experience replay. (HER)')
            # raise NotImplementedError('HER proved not viable (yet) in this dense training env.')
            # recursive for replay buffer
            return np.array([self.compute_reward(ag, dg, i) for (ag, dg, i) in zip(achieved_goal, desired_goal, info)])
            # return achieved_goal[:,0] < cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        threshold_hold = self.ep_reward_threshold
        threshold_escape = self.tr_goaldist_max

        distrecord = self.tr_multigoal_distrecords[self.fep_goalid]
        meandist = achieved_goal[0] / self.ep_goaldist_mean if self.ep_goaldist_mean > 0 else 0
        meandist = np.clip(meandist, -1, 2)
        meandist_inv = 1 - meandist

        if achieved_goal[0] <= threshold_hold:
            reward = 1 # yes

            if np.all(achieved_goal[1:] > 0):
            # if np.mean(achieved_goal[1:]) > 0:
                reward = 0

        elif achieved_goal[0] <= threshold_escape:
            reward = 0

            # all() vs. any()
            # dont always look on the compass (else dependency/overfit) - only every k episode? less and less? (decaying)
            # NN learns to follow/"feel" compass other than rely on positional obs (ie. in sparse mode), if derivative compass data is in obs/observed?! (positional overfit minimized (eliminated?): new (goal) generality level)
            if self.fep_is_dense: # compass, else sparse
                if np.all(achieved_goal[1:] < 0):
                    reward = 1
                #     reward = (1 + meandist_inv) # closer -> larger

                if np.all(achieved_goal[1:] > 0): # risk of unprecision ('last mile')
                # if np.mean(achieved_goal[1:]) > 0: # risk of overfit ('good here')
                    # reward = (-(1 + meandist)) # farer -> larger
                    reward = -1

                # reward /= 2 # normalize to 0,1

        else:
            reward = 0 # -1
            LOG.warning('unhandled reward case.')

        return np.array([reward])


    def reset_model(self):
        if self.ep_current_obs and len(self.ep_goaldists) > 0:
            ep_goalzone_per_step = np.round(self.ep_num_steps_goal_zone /  self.ep_num_steps, 2)
            # episode report
            LOG.debug('fep_lives %s', self.fep_lives)
            LOG.debug('ep_num_steps %s', self.ep_num_steps)
            LOG.debug('ep_num_steps_goal_zone %s', self.ep_num_steps_goal_zone)
            LOG.debug('ep_first_reward_step %s', self.ep_first_reward_step)
            LOG.debug('ep_goaldims_active %s', np.nonzero(self.ep_goalweight)[0])
            LOG.debug('ep_goaldist_desired %s', self.ep_current_obs['desired_goal'][0])
            LOG.debug('ep_goaldist_first %s', self.ep_goaldists[0])
            LOG.debug('ep_goaldist_min %s', np.min(self.ep_goaldists))
            LOG.debug('ep_goaldist_mean %s', np.mean(self.ep_goaldists))
            LOG.debug('ep_goaldist_max %s', np.max(self.ep_goaldists))
            LOG.debug('ep_goaldist_last %s', self.ep_current_obs['achieved_goal'][0])
            LOG.debug('ep_reward_threshold %s %s', cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT, self.ep_reward_threshold)
            LOG.debug('ep_traj_is_halved %s', self.ep_traj_is_halved)
            LOG.debug('ep_rewards_mean %s', self.ep_rewards_mean)
            LOG.debug('ep_goalzone_per_step %s', ep_goalzone_per_step)
            LOG.debug('ep_goalconv_mean %s', np.mean(np.diff(self.ep_goaldists)))
            LOG.debug('\n')

            if not self.tr_is_eval and self.tr_num_steps > 10:
                performance = np.mean(self.tr_multigoal_lastmeans)
                if performance < 100:
                    self.outfile_tr_multigoal_lastmeans.write('%s\n' % (performance))
                    self.outfile_tr_multigoal_lastmeans.flush()                
                    self.outfile_tr_goalprogresses.write('%s\n' % (self.ep_goalprogress))
                    self.outfile_tr_goalprogresses.flush()


        if not self.tr_is_eval and cfg.TrajectoryHalving.IS_ENABLED:
            self.fep_lives -= 1
            if self.fep_lives > 0 and len(self.ep_states) > 2:
                SAVEPOINT_MIN_STEPS_BEFORE_TERMINATION = 100
                # SAVEPOINT_MIN_STEPS_BEFORE_TERMINATION = 0
                return self._reset_half_episode(SAVEPOINT_MIN_STEPS_BEFORE_TERMINATION, 0)

        return self._reset_full_episode()


    def _reset_full_episode(self):
        LOG.info('\nNEW GAME.')
        (init_qpos, init_qvel) = self.init_qpos, self.init_qvel
        # (init_qpos, init_qvel) = self._add_noise_to_state(init_qpos, init_qvel)
        self.set_state(init_qpos, init_qvel)
        self.ep_states.append((init_qpos, init_qvel))
        obs_init = self._get_obs()
        self.ep_dictobs.append(obs_init)

        self.fep_savepoint_steps = 0
        self.fep_savepoint_steps_goal_zone = 0
        self.fep_goaldist_init = obs_init['achieved_goal'][0]
        self.fep_goaldist_min = obs_init['achieved_goal'][0]
        self.fep_obs_init = obs_init
        # self.fep_goaldims_primary = np.random.randint(2, size=1) # multiple?
        # TODO if random, then only secondary interval?
        # self.fep_goaldims_secondary = np.random.randint(len(self.ep_goalweight), size=1) # multiple?
        self.last_ep_goaldist_min = np.inf
        self.last_ep_rewards_mean = 0
        self.ep_traj_is_halved = False
        self.fep_lives = cfg.TrajectoryHalving.MAX_LIVES
        # self.fep_is_dense = self.tr_n_feps % 2 == 0

        # TODO redo noise?
        # noisy relative threshold (varies by initial state noise)
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * obs_init['achieved_goal']
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * (self.tr_goaldist_maxs_mean - self.tr_goaldist_mins_mean)
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT
        # self.ep_goaldist_min = obs_init['achieved_goal']
        # self.ep_goaldist_max = obs_init['achieved_goal']

        self.tr_goaldist_mins_mean = ((self.tr_n_feps * self.tr_goaldist_mins_mean) + self.fep_goaldist_min) / (self.tr_n_feps + 1)
        self.tr_goaldist_maxs_mean = ((self.tr_n_feps * self.tr_goaldist_maxs_mean) + self.fep_goaldist_max) / (self.tr_n_feps + 1)
        self.tr_n_feps += 1

        LOG.debug('tr_n_feps %s', self.tr_n_feps)
        LOG.debug('tr_obsdims %s', obs_init['observation'].shape[-1])
        LOG.debug('tr_obs_min %s %s', np.min(obs_init['observation']), np.argmin(obs_init['observation']))
        LOG.debug('tr_obs_mean %s', np.mean(obs_init['observation']))
        LOG.debug('tr_obs_std %s', np.std(obs_init['observation']))
        LOG.debug('tr_obs_max %s %s', np.max(obs_init['observation']), np.argmax(obs_init['observation']))
        LOG.debug('tr_goaldist_min %s', self.tr_goaldist_min)
        LOG.debug('tr_goaldist_max %s', self.tr_goaldist_max)
        LOG.debug('tr_goaldist_mins_mean %s', self.tr_goaldist_mins_mean)
        LOG.debug('tr_goaldist_maxs_mean %s', self.tr_goaldist_maxs_mean)
        LOG.debug('tr_goalconv_min %s', self.tr_goalconv_min)
        LOG.debug('tr_goalconv_max %s', self.tr_goalconv_max)
        LOG.debug('tr_reward_min %s', self.tr_reward_min)
        LOG.debug('tr_reward_max %s', self.tr_reward_max)
        LOG.debug('tr_multigoal_lastmeans %s', self.tr_multigoal_lastmeans)
        LOG.debug('tr_multigoal_distrecords %s', self.tr_multigoal_distrecords)
        LOG.debug('tr_training_started %s', self.tr_learning_started) 
        LOG.debug('fep_goalid %s', self.fep_goalid)
        LOG.debug('fep_goalhash %s', self.fep_goalhash)
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min)
        LOG.debug('fep_goaldist_max %s', self.fep_goaldist_max)
        LOG.debug('fep_rewards_sum %s', self.fep_rewards_sum)
        LOG.debug('fep_savepoint_steps %s', self.fep_savepoint_steps)
        LOG.debug('fep_num_steps_goal_zone %s', self.fep_savepoint_steps_goal_zone + self.ep_num_steps_goal_zone)
        LOG.debug('fep_savepoint_goaldist %s', obs_init['achieved_goal'][0])
        LOG.debug('fep_is_dense %s', self.fep_is_dense)
        self._reset()

        self.fep_rewards_sum = 0

        return obs_init


    def _reset_half_episode(self, steps_before_term, steps_offset):
        idx_halving = 0

        strat = self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE

        idx_halving = self._get_idx_for_trajectory_halving(strat, steps_before_term, steps_offset)

        if idx_halving > 0: # approached
            self.fep_savepoint_steps_goal_zone += self.ep_num_steps_goal_zone
            self.fep_lives = self.cfg.TrajectoryHalving.MAX_LIVES

        qpos, qvel = self.ep_states[idx_halving]
        self.fep_savepoint_steps += idx_halving
        LOG.info('savepoint at step %s (%s) (%s) %s', self.fep_savepoint_steps, self.fep_lives, np.round(self.ep_goaldist_min, 2), strat)

        # TODO remove or add noise?
        # qpos, qvel = self._add_noise_to_state(qpos, qvel)
        self.set_state(qpos, qvel)
        self.ep_traj_is_halved = True
        self.ep_rewards_sum = 0
        self.ep_num_steps = 0
        self.ep_num_steps_goal_zone = 0
        self.ep_num_steps_conv = 0
        self.ep_last_step_approached = 0
        self.last_ep_rewards_mean = self.ep_rewards_mean
        self.last_ep_goaldist_min = self.ep_goaldist_min

        assert (len(self.ep_states)
                == len(self.ep_actions)
                == len(self.ep_rewards)
                == len(self.ep_dictobs)
                == len(self.ep_goaldists)
                == len(self.ep_goalconvs)), 'check state integrity'

        self.ep_states = [self.ep_states[idx_halving]]
        self.ep_actions = [self.ep_actions[idx_halving]]
        self.ep_rewards = [self.ep_rewards[idx_halving]]
        self.ep_dictobs = [self.ep_dictobs[idx_halving]]
        self.ep_goaldists = [self.ep_goaldists[idx_halving]]
        self.ep_goalconvs = [self.ep_goalconvs[idx_halving]]

        obs_init = self._get_obs()
        self.ep_goaldist_min = obs_init['achieved_goal'][0]
        self.ep_goaldist_max = obs_init['achieved_goal'][0]

        return obs_init


    def _reset(self):
        self.ep_goaldist_mean: float = 0
        self.ep_rewards_mean: float = 0
        self.ep_rewards_sum = 0
        self.ep_num_steps: int = 0
        self.ep_num_steps_conv: int = 0
        self.ep_last_step_approached: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_last_reward_step: int = -1
        self.ep_last_goaldiff = np.zeros(self.cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION)
        self.ep_last_obs_worldderivs = np.zeros(self.data.qpos.flatten().shape[0] * self.cfg.General.WORLD_DERIV_ORDERS)
        self.ep_dictobs = []
        self.ep_current_obs = None
        self.ep_current_reward = 0
        self.ep_goaldists = []
        self.ep_goalconvs = []
        self.ep_states = []
        self.ep_rewards = []
        self.ep_actions = []
        self.ep_goalprogress = 0
        self.ep_num_steps_goal_zone = 0
        self.ep_count_fails_pose_detection = 0
       
        if not self.landmarker_achieved:
            self.landmarker_achieved = HandLandmarker.create_from_options(self.landmarker_options_achieved)
            self.landmarker_desired = HandLandmarker.create_from_options(self.landmarker_options_desired)

        VidCapSingletonProc.change_img_ev.set()
        
        if self.tr_is_eval:
            self.fep_goalid = 0
        else:
            IS_GOAL_SAMPLING_UNIFORM = False
            # prio sampling may lead to favorism? (convergence to only single most difficult goal?)
            IS_GOAL_SAMPLING_MEAN = True
            IS_GOAL_SAMPLING_BAD = False
            IS_GOAL_SAMPLING_LAST = False
            if IS_GOAL_SAMPLING_UNIFORM:
                self.fep_goalid = np.random.randint(len(self.tr_multigoal_paths))
            elif IS_GOAL_SAMPLING_MEAN:
                lastmeans = np.array(self.tr_multigoal_lastmeans)
                # lastmeans = np.array(self.tr_multigoal_lastmeans) ** 2
                lastmeans_normed = lastmeans / np.sum(lastmeans)
                self.fep_goalid = np.random.choice(np.arange(len(self.tr_multigoal_paths)), p=lastmeans_normed)
            elif IS_GOAL_SAMPLING_BAD:
                lastmeans_based = np.array(self.tr_multigoal_lastmeans)
                lastmeans_based -= np.min(self.tr_multigoal_lastmeans)
                lastmeans_based /= np.sum(self.tr_multigoal_lastmeans)
                if np.sum(lastmeans_based) == 0:
                    self.fep_goalid = np.random.choice(np.arange(len(self.tr_multigoal_paths)))
                else:
                    self.fep_goalid = np.random.choice(np.arange(len(self.tr_multigoal_paths)), p=lastmeans_based)
            elif IS_GOAL_SAMPLING_LAST:
                self.fep_goalid = np.argmax(self.tr_multigoal_lastmeans)

            desired_imgpath = self.tr_multigoal_paths[self.fep_goalid]
            self.desired_imgdata = image.imread(desired_imgpath)


    def _get_idx_for_trajectory_halving(self, strat, steps_before_term = 10, steps_offset = -10):
        idx_step = 0
        match strat:
            case self.cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case self.cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(self.ep_goalconvs)
                if idx_step == len(self.ep_states):
                    idx_step = 0
            case self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goaldists)
            case self.cfg.TrajectoryHalving.Strat.LAST_STEP_GOAL_ZONE:
                idx_step = len(self.ep_goaldists) - np.argmax(np.array(self.ep_goaldists[::-1]) < self.ep_reward_threshold)
            case self.cfg.TrajectoryHalving.Strat.HIGHEST_REWARD:
                idx_step = np.argmax(self.ep_rewards)
            case self.cfg.TrajectoryHalving.Strat.LAST_POSITIVE_REWARD:
                idx_step = len(self.ep_rewards) - np.argmax(np.array(self.ep_rewards[::-1]) > 0)
            case self.cfg.TrajectoryHalving.Strat.LAST_POSITIVE_CONVERGENCE:
                idx_step = len(self.ep_goalconvs) - np.argmax(np.array(self.ep_goalconvs[::-1]) > 0)
        
        idx_step = min(idx_step, len(self.ep_states) - steps_before_term)
        idx_step += steps_offset
        idx_step = max(0, idx_step)
        idx_step = min(len(self.ep_states) - 1, idx_step)
        return idx_step


    def _add_noise_to_state(self, qpos, qvel):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        return qpos, qvel


    def _add_noise(self, obs, noise_low=-1e-2, noise_high=1e-2):
        obs = obs + np.random.uniform(
            low=noise_low, high=noise_high, size=obs.shape[-1]
        )
        return obs


    def _normalize_unit_limit(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        if max_val == min_val:
            return 0.5
        return (val - min_val) / (max_val - min_val)


    def _normalize_L2(self, vector):
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


def unit_vector(vector):
    return vector / np.linalg.norm(vector)
    

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_rotation_matrix(vec1, vec2):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def rotate_vector_to_match(vec_a, vec_b):
    # Normalize vectors
    a = vec_a / np.linalg.norm(vec_a)
    b = vec_b / np.linalg.norm(vec_b)

    # Compute rotation axis (cross product) and angle (dot product)
    axis = np.cross(a, b)
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

    # Handle the case when vectors are already aligned or opposite
    if np.allclose(axis, 0):  # vectors are collinear
        if np.dot(a, b) > 0:
            return vec_a  # already aligned
        else:
            # 180° rotation around any orthogonal axis
            orthogonal = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
            axis = np.cross(a, orthogonal)
            axis /= np.linalg.norm(axis)
            rot = R.from_rotvec(np.pi * axis)
            return rot.apply(vec_a)

    # Normalize axis and create rotation
    axis /= np.linalg.norm(axis)
    rot = R.from_rotvec(angle * axis)

    # Apply rotation
    return rot.apply(vec_a)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image


def parallel_plot(queue: multiprocessing.Queue):
    fig = plt.figure()
    ax2 = fig.add_subplot(211)
    plot_desired = ax2.imshow(np.zeros((1,1,3)))
    # ax1 = fig.add_subplot(132)
    # plot_achieved = ax1.imshow(np.zeros((1,1,3)))
    extplot = fig.add_subplot(212, projection="3d")

    while True:
        achieved_img, desired_img, achieved_pose, desired_pose = queue.get()

        plot_desired.set_data(desired_img)
        plot_desired.draw(plot_desired.get_figure().canvas.get_renderer())

        # plot_achieved.set_data(achieved_img)
        # plot_achieved.draw(plot_achieved.get_figure().canvas.get_renderer())

        # plot topology connection
        # https://github.com/stebusse/mediapipe-plot-pose-live/blob/main/plot_pose_live.py
        extplot.clear()
        extplot.set_xlabel('x')
        extplot.set_ylabel('z')
        extplot.set_zlabel('y')
        extplot.set_xlim3d(-0.5, 0.5)
        extplot.set_ylim3d(-0.5, 0.5)
        extplot.set_zlim3d(0.5, -0.5) # flip z-axis

        if achieved_pose.hand_landmarks:
            for group in cfg.General.groups_filtered:
                plotX = [achieved_pose.hand_landmarks[0][i].x for i in group]
                plotY = [achieved_pose.hand_landmarks[0][i].y for i in group]
                plotZ = [achieved_pose.hand_landmarks[0][i].z for i in group]
                if 1 in group: # thumb
                    extplot.plot(plotX, plotZ, plotY, color='red')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='red', marker='.', linestyle = 'dashed')

        if desired_pose.hand_landmarks:
            for group in cfg.General.groups_filtered:
                plotX = [desired_pose.hand_landmarks[0][i].x for i in group]
                plotY = [desired_pose.hand_landmarks[0][i].y for i in group]
                plotZ = [desired_pose.hand_landmarks[0][i].z for i in group]
                if 1 in group: # thumb
                    extplot.plot(plotX, plotZ, plotY, color='green')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='green', marker='.', linestyle = 'dashed')

        extplot.draw(extplot.get_figure().canvas.get_renderer())
        plt.pause(0.00001)


def vector_to_uniform_scalar(vector, base=256):
    """
    index-based "hash" (positional encoding)
    base >= vector_length
    """
    # Convert vector to unique scalar using base conversion
    scalar = 0
    for i, val in enumerate(reversed(vector)):
        scalar += val * (base ** i)

    # Normalize scalar to [0, 1] with uniform step
    max_val = base ** len(vector) - 1
    if max_val == 0:
        return 1
    else:
        return scalar / max_val


def trunc(vals, decs=0):
    return np.trunc(vals*10**decs)/(10**decs)


# Decorrelating loss
def decorrelation_loss(z):
    # z: [batch_size, latent_dim]
    z = z - z.mean(dim=0, keepdim=True)  # zero mean
    cov = (z.T @ z) / (z.shape[0] - 1)   # covariance matrix
    diag = torch.diag(cov)
    off_diag = cov - torch.diag_embed(diag)
    return (off_diag**2).sum()  # penalize off-diagonal terms


def normalized_angle(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    # Mask to exclude NaNs from both vectors
    valid_mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(valid_mask):
        return 0  # No valid data

    a_valid = a[valid_mask]
    b_valid = b[valid_mask]
    norm_a = np.linalg.norm(a_valid)
    norm_b = np.linalg.norm(b_valid)
    if norm_a == 0 or norm_b == 0:
        return 0  # Cannot compute angle with zero-length vector

    dot_product = np.dot(a_valid, b_valid)
    cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return angle_rad / np.pi  # Normalized to [0, 1]