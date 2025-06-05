
import numpy as np
import time
import logging
import git
import uuid
import chromadb
from types import SimpleNamespace
from . import pose_imitation_cfg as cfg
from gymnasium import spaces

import multiprocessing
from multiprocessing.queues import Empty
import matplotlib.pyplot as plt
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

BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker

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

# https://cookbook.chromadb.dev/running/performance-tips/#__tabbed_1_1


ACTION_OBS_IS_ENABLED = False



# 1M, convRewarding, groundContactTerm., metaGoals0.5, threshold0.05:  converging, no pleateaus yet /home/t14/Documents/tuhh/dsf/Scilab-RL/data/053b120/le-pose-imitation-v4/10-43-06/rl_model_finished
# 1M(!!!!), posConvRewardingOnly, no goalzone, meanTermPen: clear converging, no plateau yet restore_policy=/home/t14/Documents/tuhh/dsf/Scilab-RL/data/7aca00b/le-pose-imitation-v4/18-52-08/rl_model_finished

# 0.1M, convRewMagNorm, no HER:     uses some momentum /home/t14/Documents/tuhh/dsf/Scilab-RL/data/62fb159/le-pose-imitation-v4/13-13-53/rl_model_finished
# 0.1M, convRewMagNorm, HER:    uses arms to support /home/t14/Documents/tuhh/dsf/Scilab-RL/data/62fb159/le-pose-imitation-v4/13-13-53/rl_model_finished

# TODO cleansac#296: add locality propagation exps. to HER?
# TODO obs appender func with limits warning (for normalization(!))
class PoseImitationEnv(HumanoidEnv):


    def __init__(self, is_eval=False, is_render=True, log_level=logging.INFO):
        LOG.setLevel(log_level)

        HumanoidEnv.__init__(self,
                             exclude_current_positions_from_observation=True,
                             width=cfg.General.RENDER_IMAGE_SIZE,
                             height=cfg.General.RENDER_IMAGE_SIZE,
                             xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/humanoid.xml')
                            #  xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/robotis_op3/scene.xml')
                            #  xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/agility_cassie/scene.xml')
        self.frame_skip: 5 = cfg.General.FRAMESKIP_STEP

        assert cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT and cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT, 'cannot plot in a step with no pose render (and detection'

        self.cfg = cfg
        self.is_render = is_render
        self.is_eval = is_eval
        self.outfile_fep_num_steps_goal_zone = open('fep_num_steps_goal_zone.dat', 'a')

        img_array = image.imread(PATH_GIT_WORKING_DIR + '/mediapipe/poses/pose2.jpg')
        img_array = image.imread(PATH_GIT_WORKING_DIR + '/mediapipe/poses/pose1.jpg')
        self.desired_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array.copy())
        self.desired_pose = None

        # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
        self.landmarker_options_achieved = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=PATH_GIT_WORKING_DIR + '/mediapipe/model/pose_landmarker_full.task',            
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
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5)
        self.landmarker_achieved = PoseLandmarker.create_from_options(self.landmarker_options_achieved)

        self.landmarker_options_desired = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=PATH_GIT_WORKING_DIR + '/mediapipe/model/pose_landmarker_lite.task',            
                delegate=BaseOptions.Delegate.GPU),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1)
        self.landmarker_desired = PoseLandmarker.create_from_options(self.landmarker_options_desired)

        obspace_total_dims = 0
        if ACTION_OBS_IS_ENABLED:
            obspace_total_dims += self.action_space.shape[0] # action

        # world obs
        obspace_total_dims += self.observation_space.shape[0] # super
        obspace_total_dims += self.data.qpos.shape[0] * self.cfg.General.OBS_WORLD_DERIV_ORDERS # superpos-diffs

        # achieved obs
        obspace_total_dims += 2 # achieved: velo-z, height
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # achieved: pose

        # desired obs
        obspace_total_dims += 2 # desired: velo-z, height
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # desired: pose

        # goal obs
        obspace_total_dims += 1 + self.cfg.General.GOAL_DERIV_ORDERS

        # reward obs
        obspace_total_dims += 1 + self.cfg.General.OBS_REWARD_HISTORY_LENGTH
        obspace_total_dims += self.cfg.General.REWARD_DERIV_ORDERS



        # if self.cfg.MetaObservation.IS_ENABLED:
            # obspace_total_dims += 3 # goaldist, goal_convergence, is_seeking_goal
            # obspace_total_dims += 2 # desired: velo-z, height
        
        # if self.cfg.DbObservation.IS_ACTIONDB_ENABLED:
        #     obspace_total_dims += 2 # best_similarity, best_reward
        #     obspace_total_dims += 20 # best_action

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
        self.init_qpos[6] = -1.4 # face towards camera
        self.tr_feps_total = 0
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
        self.ep_goaldist_min: float = np.inf
        self.ep_goaldist_max: float = 0
        self.ep_goalweight = []
        self.ep_lives = cfg.TrajectoryHalving.MAX_LIVES
        self.lp_num_steps = 0

        # constant threshold
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self.landmarker_achieved = None
        self.landmarker_desired = None
        self.last_ob_pose_achieved = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)
        self.last_ob_pose_desired = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)

        # vecDB
        chroma_client = chromadb.EphemeralClient()
        if is_eval:
            # https://cookbook.chromadb.dev/core/collections/#collection-properties
            self.actiondb = chroma_client.create_collection(name='obs_eval', metadata={'hnsw:space': 'cosine'}) # l2, cosine, ip
        else:
            self.actiondb = chroma_client.create_collection(name='obs_train', metadata={'hnsw:space': 'cosine'})

        # if self.is_plot:
        #     self.parallel_plot_queue = multiprocessing.Queue()
        #     multiprocessing.log_to_stderr(logging.DEBUG)
        #     multiprocessing.Process(target=parallel_plot, args=((self.parallel_plot_queue,)), daemon=True).start()

        self._reset()
        LOG.debug('le-walker-2d initialized.')


    def step(self, action):
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
        goalacce = obs['achieved_goal'][2]
        self.ep_goaldists.append(goaldist)
        self.ep_goalconvs.append(goalconv)
        self.ep_goalacces.append(goalacce)
        self.ep_dictobs.append(obs)
        self.ep_current_obs = obs

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info).item()
        self.ep_rewards.append(reward)

        if self.cfg.DbObservation.IS_ACTIONDB_ENABLED:
            # currently: adding db-obs worsens performance/rewards
            # TODO polluted: keep only the best of the best! (needs cleaning/denoising/filtering) ("king-of-the-canyon")
            actiondb_best_embedding = obs['observation'].dtype.metadata['actiondb_best_embedding']
            actiondb_best_id = obs['observation'].dtype.metadata['actiondb_best_id']
            actiondb_best_reward = obs['observation'].dtype.metadata['actiondb_best_reward']
            actiondb_best_similarity = obs['observation'].dtype.metadata['actiondb_best_similarity']
            self.ep_actiondb_similarity_mean = (((self.tr_num_steps - 1) * self.ep_actiondb_similarity_mean) + actiondb_best_similarity) / (self.tr_num_steps)

        # records                    
        if goaldist < self.tr_goaldist_min:
            LOG.debug(f"TR IMPROVED: {goaldist} < {self.tr_goaldist_min}")
            self.tr_goaldist_min = goaldist

        if goaldist > self.tr_goaldist_max:
            LOG.debug(f"TR DEPROVED: {goaldist} > {self.tr_goaldist_max}")
            self.tr_goaldist_max = goaldist        

        if goaldist < self.ep_goaldist_min:
            self.ep_goaldist_min = goaldist
            self.ep_lives = cfg.TrajectoryHalving.MAX_LIVES

        if goaldist > self.ep_goaldist_max:
            self.ep_goaldist_max = goaldist

        if goaldist < self.fep_goaldist_min:
            self.fep_goaldist_min = goaldist

        if goaldist > self.fep_goaldist_max:
            self.fep_goaldist_max = goaldist

        if goalconv < self.tr_goalconv_min:
            self.tr_goalconv_min = goalconv

        if goalconv > self.tr_goalconv_max:
            self.tr_goalconv_max = goalconv

        if reward < self.tr_reward_min:
            self.tr_reward_min = reward

        if reward > self.tr_reward_max:
            self.tr_reward_max = reward

        if self.ep_num_steps > self.tr_ep_num_steps_max:
            self.tr_ep_num_steps_max = self.ep_num_steps

        if goaldist < self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT:
            self.ep_num_steps_goal_zone += 1


        terminated = False
        truncated = False

        # space constraint
        # reckless training (no penalties, fast respawn)
        if self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE and self.ep_num_steps > self.cfg.PracticeSpace.STEPS_INVINCIBLE_SPAWN:

            MAX_DIVERGENT_STEPS = 500 # 75
            # BORDER_HEIGHT_MIN = 0.2 # op3
            BORDER_HEIGHT_MIN = 0.7 # gym-humanoid

            if self.ep_rewards_sum < -30:
                terminated = True

            # if reward <= 0:
            #     terminated = True

            # # TODO only in goal-hold phase? (goal-reach may need divergent steps...)
            # elif len(self.ep_goalconvs) > MAX_DIVERGENT_STEPS and not np.argmax(np.array(self.ep_goalconvs[-MAX_DIVERGENT_STEPS:]) > 0):
            #     terminated = True
            #     # reward = -1
            #     LOG.info('TOO MANY CONSEQUENT DIVERGENT STEPS.')

            elif self.data.qpos[2] < BORDER_HEIGHT_MIN:  # practice height (tight limit for efficiency?)
                terminated = True
                # reward = -1
                LOG.info('HEIGHT TOO LOW/HIGH. %s %s', reward, self.ep_rewards_sum)


            # min. convergence terminate? ("flaming wall")
            
            # elif self.ep_count_fails_pose_detection > 10:
            #     LOG.info('TOO MANY DETECTION FAILURES. (better detection at higher res.?)')
            #     terminated = True
            #     self.ep_lives -= 1
            #     reward = 0


        self.ep_rewards_mean = (((self.ep_num_steps - 1) * self.ep_rewards_mean) + reward) / (self.ep_num_steps)

        # also skip first buggy render
        if self.tr_feps_total == 1 or self.ep_num_steps > self.cfg.General.MAX_STEPS_EPISODE_TRUNCATION:
            LOG.info('TRUNCATED.')
            truncated = True
            is_success = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            info['success'] = is_success

        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(np.round(reward, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'ep_rewards_mean', str(np.round(self.ep_rewards_mean, 2)))
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'goaldist', str(np.round(goaldist, 2)))
            fep_num_steps_goal_zone = self.fep_savepoint_steps_goal_zone + self.ep_num_steps_goal_zone
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'fep_num_steps_goal_zone', str(fep_num_steps_goal_zone))
            human_viewer.render()

        self.ep_current_reward = reward
        self.ep_rewards_sum += reward
        self.fep_rewards_sum += reward

        result = obs, reward, terminated, truncated, info
        return result


    # obs = achieved_obs + metaobs
    # TODO keep obs keys/indices mapping (eg. dict, vs. "counting")
    def _get_obs(self):
        obs = np.array([])

        # =========== WORLD OBS

        obs_world = np.array([])

        # stabilizer
        # ob_achieved_primary_height = self._normalize_unit_limit(self.data.qpos[2], 0.0, 0.3) # op3
        # obs_world = np.append(obs_world, ob_achieved_primary_height)

        # ob_primary_velo_head = np.sqrt(np.square(self.data.qvel[0]) + np.square(self.data.qvel[1]) + np.square(self.data.qvel[2]))
        # obs_world = np.append(obs_world, ob_primary_velo_head)

        # ob_primary_acc_head = self._normalize_unit_limit(self.data.sensor('head_acc_sensor').data, -50, 50) # op3
        # obs_world = np.append(obs_world, ob_primary_acc_head) # 3

        # ob_primary_l_foot_touch = self._normalize_unit_limit(self.data.sensor('l_foot_touch_sensor').data, 0, 100) # op3
        # obs_world = np.append(obs_world, ob_primary_l_foot_touch)

        # ob_primary_r_foot_touch = self._normalize_unit_limit(self.data.sensor('r_foot_touch_sensor').data, 0, 100) # op3
        # obs_world = np.append(obs_world, ob_primary_r_foot_touch)

        obs_world = np.append(obs_world, super()._get_obs()) # already includes first order (mujoco-computed, possibly different)

        worldderiv_orders = self.cfg.General.OBS_WORLD_DERIV_ORDERS
        if worldderiv_orders > 0:
            joint_posis = np.array([q[0] for q in self.ep_states[-(2 ** worldderiv_orders):]]) # only enough recent posis for all orders (2^k)
            joint_posis = np.pad(joint_posis, ((2 ** worldderiv_orders,0), (0,0))) # pad for always enough recent posis

            for i in range(1, worldderiv_orders + 1):
                obs_world = np.append(obs_world, np.diff(joint_posis, n=i, axis=0)[-1])

        obs = np.append(obs, obs_world)


        # needs denoising for (near-)linearity in NN
        # desired_pose = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        # achieved_pose = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        # # detect pose(s) only every nth step, else use last valid
        # if not self.desired_pose or self.ep_num_steps % cfg.General.STEPSKIP_DETECT == 0:
        #     # renders only rgb (cant render multiple modes simultanously)
        #     render_tmp = self.render_mode
        #     self.render_mode = 'rgb_array'

        #     # https://github.com/jurgisp/memory-maze/issues/26
        #     achieved_img = self.render().copy() # MUJOCO_GL=glfw
        #     self.render_mode = render_tmp
        #     achieved_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=achieved_img)
        #     # TODO get desired img from video?
        #     desired_img = self.desired_img

        #     # bottleneck start
        #     # t = time.perf_counter()
        #     # https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker#detect_for_video
        #     # video_timestamp_ms = int(time.process_time_ns() / 1000 + self.ep_num_steps)
        #     # achieved_pose = self.landmarker_achieved.detect_for_video(achieved_img, video_timestamp_ms)

        #     if not self.desired_pose:
        #         # only once at the beginning (still image)
        #         self.desired_pose = self.landmarker_desired.detect(desired_img)
        #     # LOG.debug(time.perf_counter() - t)
        #     # bottleneck end

        # desired_pose = self.desired_pose


        # =========== ACTION OBS

        if ACTION_OBS_IS_ENABLED:
            if len(self.ep_actions) >= 2:
                obs = np.append(obs, self.ep_actions[-2]) # prev action
            else:
                obs = np.append(obs, np.zeros(self.action_space.shape)) # no/first action


        
        # ========= ACHIEVED OBS

        obs_achieved = np.array([])

        # velo-z
        # ob_achieved_velo_z = self._normalize_unit_limit(self.data.qvel[2], -1, 1)
        # obs_achieved = np.append(obs_achieved, ob_achieved_velo_z)

        head_velo_x = self._normalize_unit_limit(self.data.qvel[0], -1, 1)
        head_velo_y = self._normalize_unit_limit(self.data.qvel[1], -1, 1)
        head_velo_z = self._normalize_unit_limit(self.data.qvel[2], -1, 1)
        ob_achieved_velo = np.abs(np.mean([head_velo_x, head_velo_y, head_velo_z]))
        obs_achieved = np.append(obs_achieved, ob_achieved_velo)

        ob_achieved_height = self._normalize_unit_limit(self.data.qpos[2], 0.0, 2.0) # gym-humanoid
        # ob_achieved_height = self._normalize_unit_limit(self.data.qpos[2], 0.0, 0.3) # op3
        obs_achieved = np.append(obs_achieved, ob_achieved_height) # ob_primary_height


        # obs_achieved = np.append(obs_achieved, obs[1]) # ob_primary_velo_head

        # if desired_pose.pose_world_landmarks:
        #     achieved_pose = copy.deepcopy(desired_pose)
        #     # body
        #     # TODO extend/unite with geom?
        #     for i, body_id in enumerate(cfg.General.MJBODY_TO_MPPOSE):
        #         if body_id:
        #             achieved_pose.pose_world_landmarks[0][i].x = self.data.body(body_id).xpos[0] * 3.6
        #             achieved_pose.pose_world_landmarks[0][i].z = self.data.body(body_id).xpos[1] * 3.6
        #             achieved_pose.pose_world_landmarks[0][i].y = -self.data.body(body_id).xpos[2] * 3.6 + 0.85
        #         else:
        #             achieved_pose.pose_world_landmarks[0][i].x = -1
        #             achieved_pose.pose_world_landmarks[0][i].y = -1
        #             achieved_pose.pose_world_landmarks[0][i].z = -1

        # achieved_obs = np.append(achieved_obs, achieved_ob_pose)

        obs = np.append(obs, obs_achieved)


        # ========= DESIRED OBS

        obs_desired = np.array([])

        ob_desired_velo_z = 0.5
        obs_desired = np.append(obs_desired, ob_desired_velo_z) # velo-z

        ob_desired_height = self._normalize_unit_limit(1.4, 0.0, 2.0) # gym-humanoid
        # ob_desired_height = self._normalize_unit_limit(0.3, 0.0, 0.3) # height (op3)
        obs_desired = np.append(obs_desired, ob_desired_height)

        # ob_desired_primary_velo_head = 0
        # obs_desired = np.append(obs_desired, ob_desired_primary_velo_head)

        # desired_ob_pose = self.last_ob_pose_desired
        # # desired_ob_pose = self.last_ob_pose_desired
        # if desired_pose.pose_world_landmarks:
        #     # only first detected pose
        #     # desired_ob_pose = [(landmark.x, landmark.y - 1.2, landmark.z) for landmark in desired_pose.pose_world_landmarks[0]]
        #     desired_ob_pose = [(landmark.x, landmark.y, landmark.z) for landmark in desired_pose.pose_world_landmarks[0]]
        #     desired_ob_pose = np.array(desired_ob_pose)[cfg.General.IDS_LANDMARKS_FILTERED]
        #     desired_ob_pose = self._normalize_to_limits(desired_ob_pose, -1, 1)
        #     # TODO check if detected pose is valid/possible (height, change, confidence etc.)
        #     self.last_ob_pose_desired = desired_ob_pose
        # desired_obs = np.append(desired_obs, desired_ob_pose)



        # ========= GOAL

        # combing? (stepwise-combing not working with goalconv-rewards(prev. step goal differs))
        self.ep_goalweight = np.full(obs_desired.shape, 1.0)
        # self.ep_goalweight[0] = 1 # base primary dim
        # self.ep_goalweight[1] = 1 # base primary dim
        # goaldims_primary = np.random.randint(2, size=1) # multiple?
        # self.ep_goalweight[goaldims_primary] = 1
        # self.ep_goalweight[goaldims_secondary] = 0.5 # never abandon primary goal in favor of secondary goals
        goaldiff_weighted = self.ep_goalweight * (obs_achieved - obs_desired)
        goaldist = np.linalg.norm(goaldiff_weighted, axis=-1)

        goalderiv_orders = self.cfg.General.GOAL_DERIV_ORDERS
        goalderivs = np.array([])
        if goalderiv_orders > 0:
            goaldists = np.array(self.ep_goaldists)
            goaldists = np.append(goaldists, goaldist) # most recent
            goaldists = np.array(goaldists[-(2 ** goalderiv_orders):]) # only enough recent goaldists for all orders (2^k)
            goaldists = np.pad(goaldists, (2 ** goalderiv_orders,0)) # pad for more than enough recents

            for i in range(1, goalderiv_orders + 1):
                goalderivs = np.append(goalderivs, np.diff(goaldists, n=i, axis=0)[-1])

        obs = np.append(obs, obs_desired.ravel()) # goal
        obs = np.append(obs, goaldist)
        obs = np.append(obs, goalderivs)


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
    

            # obs = obs.astype(np.dtype(float, metadata={
            #     'reward': reward,
            #     }))


        # ========= DB ACTION OBS

        if self.cfg.DbObservation.IS_ACTIONDB_ENABLED:
            # find best action for current state
            best_id = None
            best_similarity = 0
            best_reward = 0
            best_action = np.zeros(self.action_space.shape)

            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            best_embedding = self._normalize_L2(np.hstack((qpos, qvel)))


            db_query = self.actiondb.query(
                query_embeddings=[best_embedding],
                n_results=1,
                # TODO also filter by goaldist?
                # performance worsens gradually with db size
                # where={'reward': {'$gt': reward_current}},
            )

            if len(db_query['ids'][0]) > 0:
                # db action found
                best_id = db_query['ids'][0][0]
                best_similarity = db_query['distances'][0][0]
                best_action = np.fromstring(db_query['documents'][0][0].strip('[]'), sep=',')
                best_reward = db_query['metadatas'][0][0]['reward']

                # if best_similarity < 0.01 and best_reward > reward_current:
                #     LOG.debug('db action is better than chosen action: %s > %s (%s)', best_reward, reward_current, best_similarity)

            if reward > 0 and len(self.ep_states) >= 2:
                # save chosen action for prev state to actiondb
                (prev_qpos, prev_qvel) = self.ep_states[-2]
                self.actiondb.add(
                    embeddings=[np.hstack((prev_qpos, prev_qvel))],
                    documents=[np.array2string(self.ep_actions[-2], separator=',', precision=16)],
                    # https://cookbook.chromadb.dev/faq/#large-distances-in-search-results
                    ids=[uuid.uuid4().hex],
                    metadatas=[{
                        "reward": reward,
                        "goaldist": goaldist,
                        "goalconv": goalderivs[0],
                        }]
                )

                    # if reward > actiondb_best_reward and actiondb_best_similarity < cfg.DbObservation.ACTIONDB_SIMILARITY_THRESHOLD:
                        # # better action in proximity: replace neighbor
                        # if actiondb_best_id:
                        #     LOG.info('DB IMPROVED. %s > %s', reward, actiondb_best_reward)

                        #     db_query = self.actiondb.query(
                        #         include=['distances'],
                        #         query_embeddings=[actiondb_best_embedding],
                        #         n_results=100,
                        #     )

                        #     ids = np.array(db_query['ids'][0])
                        #     dists = np.array(db_query['distances'][0])
                        #     self.actiondb.delete(
                        #         ids=ids[np.argwhere(dists < cfg.DbObservation.ACTIONDB_SIMILARITY_THRESHOLD)].ravel().tolist()
                        #     )


            # too much context necessary?
            obs = np.append(obs, best_similarity)
            obs = np.append(obs, best_reward)
            # directed: action? reward-/similarity-scaled action? (similarity-scaled) diff between chosen action and best action? 
            obs = np.append(obs, best_similarity * self._normalize_unit_limit(best_action, -np.pi, np.pi))

            # https://stackoverflow.com/questions/67509913/add-an-attribute-to-a-numpy-array-in-runtime
            obs = obs.astype(np.dtype(float, metadata={
                'actiondb_best_embedding': best_embedding,
                'actiondb_best_id': best_id,
                'actiondb_best_reward': best_reward,
                'actiondb_best_similarity': best_similarity,
                }))

        # metagoal(s) only
        # achieved_goal = 0.5 * goaldist + 0.5 * goalconv
        # desired_goal = self.ep_reward_threshold

        achieved_goal = np.array([goaldist] + goalderivs.tolist())
        desired_goal = np.zeros(achieved_goal.shape)  # or arbitrary big numbers for max.??
        dictobs = dict(
            observation=obs,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )

        # if self.is_plot and self.ep_num_steps % cfg.General.STEPSKIP_PLOT == 0:
        #     achieved_img_annotated = draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
        #     desired_img_annotated = draw_landmarks_on_image(desired_img.numpy_view(), desired_pose)
        #     if self.parallel_plot_queue.empty():
        #         self.parallel_plot_queue.put_nowait((achieved_img_annotated, desired_img_annotated, achieved_pose, desired_pose))

        return dictobs
    

    # class Record:
    #     def __init__(self):
    #         record_min = np.inf
    #         record_max = -np.inf

    #     def get_min(self):
    #         return self.record_min


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        if achieved_goal.ndim > 1:
            # TODO possibly only after reaching goalzone? (in sparse mode)
            # raise NotImplementedError('HER proved not viable (yet) in this dense training env.')
            # recursive for replay buffer
            return np.array([self.compute_reward(ag, dg, i) for (ag, dg, i) in zip(achieved_goal, desired_goal, info)])
            # return achieved_goal[:,0] < cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT 

        diff_orders = self.cfg.General.GOAL_DERIV_ORDERS
        is_rewarding_binary = self.cfg.General.IS_REWARDING_BINARY

        # inside goaldist
        if achieved_goal[0] <= cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT:
            reward = 0 # idle

            for k in range(1, diff_orders):
                if achieved_goal[k] > 0 and achieved_goal[k+1] > 0: # effort
                    reward = 1 if is_rewarding_binary else 1/k
                    break

        # outside goaldist
        elif achieved_goal[0] > cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT:
            reward = -1 # burn (-1) vs. idle (0)

            for k in range(1, diff_orders):
                if achieved_goal[k] > 0 and achieved_goal[k+1] > 0: # escape
                    reward = 1 if is_rewarding_binary else 1/k
                    break

        return np.array([reward])


    def reset_model(self):
        if self.ep_current_obs and len(self.ep_goaldists) > 0:
            # episode report
            LOG.debug('ep_lives %s', self.ep_lives)
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
            LOG.debug('ep_goalzone_per_step %s', np.round(self.ep_num_steps_goal_zone /  self.ep_num_steps, 2))
            LOG.debug('ep_goalconv_mean %s', np.mean(np.diff(self.ep_goaldists)))
            LOG.debug('ep_actiondb_similarity_mean %s', self.ep_actiondb_similarity_mean)
            LOG.debug('\n')

            if not self.is_eval and self.tr_num_steps > 10:
                fep_num_steps_goal_zone = self.fep_savepoint_steps_goal_zone + self.ep_num_steps_goal_zone
                self.outfile_fep_num_steps_goal_zone.write('%s\n' % (fep_num_steps_goal_zone))
                self.outfile_fep_num_steps_goal_zone.flush()

        if not self.is_eval and cfg.TrajectoryHalving.IS_ENABLED:
            self.ep_lives -= 1
            if self.ep_lives > 0 and len(self.ep_states) > 2:
                SAVEPOINT_MIN_STEPS_BEFORE_TERMINATION = 100
                return self._reset_half_episode(SAVEPOINT_MIN_STEPS_BEFORE_TERMINATION, 0)

        return self._reset_full_episode()


    def _reset_full_episode(self):
        LOG.info('\nNEW GAME.')
        (init_qpos, init_qvel) = self.init_qpos, self.init_qvel
        # (init_qpos, init_qvel) = self._add_noise(init_qpos, init_qvel)
        self.set_state(init_qpos, init_qvel)
        self.ep_states.append((init_qpos, init_qvel))
        obs_init = self._get_obs()
        self.ep_dictobs.append(obs_init)

        self.fep_savepoint_steps = 0
        self.fep_savepoint_steps_goal_zone = 0
        self.fep_goaldist_init = obs_init['achieved_goal'][0]
        self.fep_goaldist_min = obs_init['achieved_goal'][0]
        self.fep_obs_init = obs_init
        self.fep_goaldims_primary = np.random.randint(2, size=1) # multiple?
        # TODO if random, then only secondary interval?
        self.fep_goaldims_secondary = np.random.randint(len(self.ep_goalweight), size=1) # multiple?
        # self.desired_obs = self._get_desired_obs()
        self.last_ep_goaldist_min = np.inf
        self.last_ep_rewards_mean = 0
        self.ep_traj_is_halved = False
        self.ep_lives = cfg.TrajectoryHalving.MAX_LIVES
        # TODO redo noise?
        # noisy relative threshold (varies by initial state noise)
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * obs_init['achieved_goal']
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * (self.tr_goaldist_maxs_mean - self.tr_goaldist_mins_mean)
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT
        # self.ep_goaldist_min = obs_init['achieved_goal']
        # self.ep_goaldist_max = obs_init['achieved_goal']

        self.tr_goaldist_mins_mean = ((self.tr_feps_total * self.tr_goaldist_mins_mean) + self.fep_goaldist_min) / (self.tr_feps_total + 1)
        self.tr_goaldist_maxs_mean = ((self.tr_feps_total * self.tr_goaldist_maxs_mean) + self.fep_goaldist_max) / (self.tr_feps_total + 1)
        self.tr_feps_total += 1

        LOG.debug('tr_feps_total %s', self.tr_feps_total)
        LOG.debug('tr_obsdims %s', obs_init['observation'].shape[-1])
        LOG.debug('tr_obs_min %s %s', np.min(obs_init['observation']), np.argmin(obs_init['observation']))
        LOG.debug('tr_obs_mean %s', np.mean(obs_init['observation']))
        LOG.debug('tr_obs_max %s %s', np.max(obs_init['observation']), np.argmax(obs_init['observation']))
        LOG.debug('tr_goaldist_min %s', self.tr_goaldist_min)
        LOG.debug('tr_goaldist_max %s', self.tr_goaldist_max)
        LOG.debug('tr_goaldist_mins_mean %s', self.tr_goaldist_mins_mean)
        LOG.debug('tr_goaldist_maxs_mean %s', self.tr_goaldist_maxs_mean)
        LOG.debug('tr_goalconv_min %s', self.tr_goalconv_min)
        LOG.debug('tr_goalconv_max %s', self.tr_goalconv_max)
        LOG.debug('tr_reward_min %s', self.tr_reward_min)
        LOG.debug('tr_reward_max %s', self.tr_reward_max)
        LOG.debug('tr_actiondb_size %s', self.actiondb.count())
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min)
        LOG.debug('fep_goaldist_max %s', self.fep_goaldist_max)
        LOG.debug('fep_rewards_sum %s', self.fep_rewards_sum)
        LOG.debug('fep_savepoint_steps %s', self.fep_savepoint_steps)
        LOG.debug('fep_num_steps_goal_zone %s', self.fep_savepoint_steps_goal_zone + self.ep_num_steps_goal_zone)
        LOG.debug('fep_savepoint_goaldist %s', obs_init['achieved_goal'][0])
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min)
        LOG.debug('fep_goaldist_max %s', self.fep_goaldist_max)
        self._reset()

        self.fep_rewards_sum = 0

        self.ep_goaldist_min = obs_init['achieved_goal'][0]
        self.ep_goaldist_max = obs_init['achieved_goal'][0]

        return obs_init


    def _reset_half_episode(self, steps_before_term, steps_offset):
        idx_halving = 0

        if self.ep_num_steps_goal_zone == 0:
            strat = self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE
        else:
            strat = self.cfg.TrajectoryHalving.Strat.LAST_STEP_GOAL_ZONE

        idx_halving = self._get_idx_for_trajectory_halving(strat, steps_before_term, steps_offset)

        if idx_halving > 0: # improved
            self.fep_savepoint_steps_goal_zone += self.ep_num_steps_goal_zone
            self.ep_lives = self.cfg.TrajectoryHalving.MAX_LIVES


        qpos, qvel = self.ep_states[idx_halving]
        self.fep_savepoint_steps += idx_halving
        LOG.info('savepoint at step %s (%s)', self.fep_savepoint_steps, self.ep_lives)

        # TODO remove or add noise?
        # qpos, qvel = self._add_noise(qpos, qvel)
        self.set_state(qpos, qvel)
        self.ep_traj_is_halved = True
        self.ep_rewards_sum = 0
        self.ep_num_steps = 0
        self.ep_num_steps_goal_zone = 0
        self.last_ep_rewards_mean = self.ep_rewards_mean
        self.last_ep_goaldist_min = self.ep_goaldist_min

        self.ep_states = [(qpos, qvel)]
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_dictobs = []
        self.ep_goaldists = []
        self.ep_goalconvs = []
        self.ep_goalacces = []

        obs_init = self._get_obs()
        return obs_init


    def _reset(self):
        self.ep_rewards_mean: float = 0
        self.ep_rewards_sum = 0
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_last_reward_step: int = -1
        self.ep_dictobs = []
        self.ep_current_obs = None
        self.ep_current_reward = 0
        self.ep_goaldists = []
        self.ep_goalconvs = []
        self.ep_goalacces = []
        self.ep_states = []
        self.ep_rewards = []
        self.ep_actions = []
        self.ep_num_steps_goal_zone = 0
        self.ep_count_fails_pose_detection = 0
        self.ep_actiondb_similarity_mean: float = 0

        if not self.landmarker_achieved:
            self.landmarker_achieved = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_achieved)
            self.landmarker_desired = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_desired)


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
                idx_step = len(self.ep_goaldists) - np.argmax(np.array(self.ep_goaldists[::-1]) < cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT)
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



def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def parallel_plot(queue: multiprocessing.Queue):
    fig = plt.figure()
    ax2 = fig.add_subplot(131)
    plot_desired = ax2.imshow(np.zeros((1,1,3)))
    ax1 = fig.add_subplot(132)
    plot_achieved = ax1.imshow(np.zeros((1,1,3)))
    extplot = fig.add_subplot(133, projection="3d")

    while True:
        achieved_img, desired_img, achieved_pose, desired_pose = queue.get()
    
        plot_desired.set_data(desired_img)
        plot_desired.draw(plot_desired.get_figure().canvas.get_renderer())

        plot_achieved.set_data(achieved_img)
        plot_achieved.draw(plot_achieved.get_figure().canvas.get_renderer())

        # plot topology connections
        # https://github.com/stebusse/mediapipe-plot-pose-live/blob/main/plot_pose_live.py
        extplot.clear()
        extplot.set_xlabel('x')
        extplot.set_ylabel('z')
        extplot.set_zlabel('y')
        extplot.set_xlim3d(-1, 1)
        extplot.set_ylim3d(-1, 1)
        extplot.set_zlim3d(1, -1) # flip z-axis

        if achieved_pose.pose_world_landmarks:
            for group in cfg.General.groups_filtered:
                plotX = [achieved_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [achieved_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [achieved_pose.pose_world_landmarks[0][i].z for i in group]
                if 11 in group: # right side
                    extplot.plot(plotX, plotZ, plotY, color='red')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='red', linestyle = 'dashed')

        if desired_pose.pose_world_landmarks:
            for group in cfg.General.groups_filtered:
                plotX = [desired_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [desired_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [desired_pose.pose_world_landmarks[0][i].z for i in group]
                if 11 in group: # right side
                    extplot.plot(plotX, plotZ, plotY, color='green')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='green', linestyle = 'dashed')
        
        
        extplot.draw(extplot.get_figure().canvas.get_renderer())
        plt.pause(0.00001)


# index-based "hash"
def vector_to_uniform_scalar(vector, base=256):
    # Convert vector to unique scalar using base conversion
    scalar = 0
    for i, val in enumerate(reversed(vector)):
        scalar += val * (base ** i)

    # Normalize scalar to [0, 1] with uniform steps
    max_val = base ** len(vector) - 1
    if max_val == 0:
        return 1
    else:
        return scalar / max_val


def trunc(vals, decs=0):
    return np.trunc(vals*10**decs)/(10**decs)