
import numpy as np
import time
import logging
import git
import copy
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

OBS_NORMALIZE_Z_SCORE = False


# 1M, convRewarding, groundContactTerm., metaGoals0.5, threshold0.05:  converging, no pleateaus yet /home/t14/Documents/tuhh/dsf/Scilab-RL/data/053b120/le-pose-imitation-v4/10-43-06/rl_model_finished
# 1M(!!!!), posConvRewardingOnly, no goalzone, meanTermPen: clear converging, no plateau yet restore_policy=/home/t14/Documents/tuhh/dsf/Scilab-RL/data/7aca00b/le-pose-imitation-v4/18-52-08/rl_model_finished

# TODO cleansac#296: add locality propagation exps. to HER?
class PoseImitationEnv(HumanoidEnv):


    def __init__(self, is_eval=False, is_render=True, log_level=logging.INFO):
        LOG.setLevel(log_level)

        HumanoidEnv.__init__(self,
                             exclude_current_positions_from_observation=True,
                             width=cfg.General.RENDER_IMAGE_SIZE,
                             height=cfg.General.RENDER_IMAGE_SIZE,
                             # xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/humanoid_face.xml')
                             xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/robotis_op3/scene.xml')
        self.frame_skip: 5 = cfg.General.FRAMESKIP_STEP

        assert cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT and cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT, 'cannot plot in a step with no pose render (and detection'

        self.cfg = cfg
        self.is_render = is_render
        self.is_eval = is_eval
        self.outfile_ep_rewards_mean = open('ep_rewards_mean.dat', 'a')

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
        obspace_total_dims += self.observation_space.shape[0] # super
        obspace_total_dims += 7 # height, head_velo, head acc-x, head acc-y, head acc-z, l_foot_touch, r_foot_touch
        obspace_total_dims += 2 # achieved: height, head_velo
        # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # achieved: pose

        if self.cfg.MetaObservation.IS_ENABLED:
            obspace_total_dims += 2 # desired: height, head_velo
            obspace_total_dims += 4 # goaldist, goalweight_hash, goal_convergence, is_converging
            # obspace_total_dims += cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION # desired: pose

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


        LOG.debug('observation_space %s', observation_space)
        LOG.info('action_space %s', self.action_space)
        LOG.debug('goal_space %s', goal_space)

        # once
        self.init_qpos[6] = -1.4 # face towards camera
        self.tr_feps_total = 0
        self.tr_feps_consecutive_neg = 0
        self.tr_goaldist_min: float = 1
        self.tr_goaldist_max: float = 0
        self.tr_goaldist_mins_mean: float = 0
        self.tr_goaldist_maxs_mean: float = 0
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
        self.ep_lives = cfg.General.MAX_LIVES
        self.lp_num_steps = 0

        # constant threshold
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self.landmarker_achieved = None
        self.landmarker_desired = None
        self.last_ob_pose_achieved = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)
        self.last_ob_pose_desired = np.full(cfg.General.NUM_OBSERVATION_DIMS_VISUAL_DETECTION, 1)

        # if self.is_plot:
        #     self.parallel_plot_queue = multiprocessing.Queue()
        #     multiprocessing.log_to_stderr(logging.DEBUG)
        #     multiprocessing.Process(target=parallel_plot, args=((self.parallel_plot_queue,)), daemon=True).start()

        self._reset()
        LOG.debug('le-walker-2d initialized.')




    def step(self, action):
        info = {}
        info['success'] = False

        # if self.ep_goalconvs: # stay on track (vs. explore other tracks)
        #     idx_dictobs_goalfastest = np.argmin(self.ep_goalconvs)
        #     state_fastest = self.ep_states[idx_dictobs_goalfastest]
        #     self.set_state(state_fastest[0], state_fastest[1])

        # if self.ep_dictobs and self.lp_num_steps < 10 and self.ep_num_steps % 10 == 0:
        #     # local propagation
        #     # local sample from state proximity vs. obs proximity? (obs)
        #     # local := (current - sampled) < dist (eg. goal_dist)
        #     dictobs_current = self.ep_dictobs[-1]

        #     # TODO only known/prev. local states

        #     # vs. sample (interpolate vs. simulate?) new (unknown) local obs
        #     idx_dictobs_goalnearest = np.argmin(self.ep_goaldists)
        #     dictobs_distant = self.ep_dictobs[idx_dictobs_goalnearest]

        #     grid_local_obs = np.linspace(dictobs_current['observation'], dictobs_distant['observation'], 4)
        #     obs_local = np.random.normal(dictobs_current['observation'], np.abs(grid_local_obs[2])) # TODO clip?
            
        #     local_dictobs = dict(
        #             observation=obs_local,
        #             achieved_goal=dictobs_current['achieved_goal'],
        #             desired_goal=dictobs_current['desired_goal'],
        #         )
            
        #     self.lp_num_steps += 1
        #     result = local_dictobs, self.ep_current_reward, False, False, info
        #     return result
        # else:
        #     self.lp_num_steps = 0





        # reduce action space?
        # action = np.clip(action, -0.5, 0.5)
        self.do_simulation(action, self.frame_skip)
        self.ep_num_steps += 1
        self.tr_num_steps += 1

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.ep_states.append((qpos, qvel))

        obs = self._get_obs()
        self.ep_dictobs.append(obs)
        self.ep_current_obs = obs


        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info).item()
        
        # # records
        # if obs['achieved_goal'] < self.ep_goaldist_min:
        #     self.ep_goaldist_min = obs['achieved_goal']
        #     self.ep_lives = cfg.General.MAX_LIVES

        # if obs['achieved_goal'] < self.fep_goaldist_min:
        #     self.fep_goaldist_min = obs['achieved_goal']
            
        # if obs['achieved_goal'] < self.tr_goaldist_min:
        #     LOG.debug(f"TR IMPROVED: {obs['achieved_goal']} < {self.tr_goaldist_min}")
        #     self.tr_goaldist_min = obs['achieved_goal']

        # if obs['achieved_goal'] > self.ep_goaldist_max:
        #     self.ep_goaldist_max = obs['achieved_goal']
            
        # if obs['achieved_goal'] > self.fep_goaldist_max:
        #     self.fep_goaldist_max = obs['achieved_goal']
            
        # if obs['achieved_goal'] > self.tr_goaldist_max:
        #     LOG.debug(f"TR DEPROVED: {obs['achieved_goal']} > {self.tr_goaldist_max}")
        #     self.tr_goaldist_max = obs['achieved_goal']

        if self.ep_num_steps > self.tr_ep_num_steps_max:
            self.tr_ep_num_steps_max = self.ep_num_steps


        terminated = False
        truncated = False

        if reward > 0:
            reward *= (self.data.qpos[2] - 0.20) # times distance to border ("far pleases less")
        else:
            reward *= self.ep_goaldists[-1] # times distance to goal ("far hurts more")

        # space constraint
        # reckless training (no penalties, fast respawn)
        if self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE and self.ep_num_steps > self.cfg.PracticeSpace.STEPS_INVINCIBLE_SPAWN:

            if self.data.qpos[2] < 0.20 or self.data.qpos[2] > 0.35:  # practice height (tight limit for efficiency?)
                # reward = -self.ep_rewards_mean
                LOG.info('HEIGHT TOO LOW/HIGH. %s', self.ep_rewards_sum)
                # terminated = True
                # self.ep_lives -= 1

                if self.is_eval:
                    terminated = True

                # respawn (lighter, instead of heavy terminate/reset)
                elif len(self.ep_goaldists) > 2:        
                    self._reset_half_episode()

            # min. convergence terminate? ("flaming wall")
            
            # elif self.ep_count_fails_pose_detection > 10:
            #     LOG.info('TOO MANY DETECTION FAILURES. (better detection at higher res.?)')
            #     terminated = True
            #     self.ep_lives -= 1
            #     reward = 0


        self.ep_rewards_mean = (((self.ep_num_steps - 1) * self.ep_rewards_mean) + reward) / (self.ep_num_steps)

        # also skip first buggy render
        if self.tr_feps_total == 1 or self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            LOG.info('TRUNCATED.')
            truncated = True
            is_success = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            info['success'] = is_success

        if self.is_render:
            self.render_mode = 'human'
            human_viewer = self.mujoco_renderer._get_viewer('human')
            human_viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'reward', str(reward))
            human_viewer.render()

        self.ep_current_reward = reward
        self.ep_rewards_sum += reward
        self.fep_rewards_sum += reward

        result = obs, reward, terminated, truncated, info
        return result


    # obs = achieved_obs + metaobs
    # TODO keep obs keys/indices mapping (eg. dict, vs. "counting")
    def _get_obs(self):

        # =========== OBS

        obs = np.array([])

        # stabilizer
        ob_primary_velo_head = np.sqrt(np.square(self.data.qvel[0]) + np.square(self.data.qvel[1]) + np.square(self.data.qvel[2]))
        obs = np.append(obs, ob_primary_velo_head)

        ob_primary_height = self._normalize_to_limits(self.data.qpos[2], 0.0, 0.3) # op3
        obs = np.append(obs, ob_primary_height)

        ob_primary_acc_head = self._normalize_to_limits(self.data.sensor('head_acc_sensor').data, -50, 50) # op3
        obs = np.append(obs, ob_primary_acc_head)

        ob_primary_l_foot_touch = self._normalize_to_limits(self.data.sensor('l_foot_touch_sensor').data, 0, 100) # op3
        obs = np.append(obs, ob_primary_l_foot_touch)

        ob_primary_r_foot_touch = self._normalize_to_limits(self.data.sensor('r_foot_touch_sensor').data, 0, 100) # op3
        obs = np.append(obs, ob_primary_r_foot_touch)

        obs = np.append(obs, super()._get_obs())



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


        # ========= DESIRED OBS

        desired_obs = np.array([])

        desired_ob_primary_velo_head = 0
        desired_obs = np.append(desired_obs, desired_ob_primary_velo_head)

        # # # desired_ob_primary_height = self._normalize_to_limits(1.4, 0.0, 2.0) # gym-humanoid
        desired_ob_primary_height = self._normalize_to_limits(0.3, 0.0, 0.3) # op3
        desired_obs = np.append(desired_obs, desired_ob_primary_height)

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


        # ========= ACHIEVED OBS

        achieved_obs = np.array([])

        achieved_obs = np.append(achieved_obs, obs[0]) # ob_primary_velo_head
        achieved_obs = np.append(achieved_obs, obs[1]) # ob_primary_height

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

        obs = np.append(obs, achieved_obs)


        # ========= GOAL

        self.ep_goalweight = np.full(desired_obs.shape, 0.0)

        self.ep_goalweight[0] = 1 # base primary dim (velocity_head)
        self.ep_goalweight[1] = 1 # base primary dim (height)

        # self.ep_goalweight[self.fep_goaldims_secondary] = 0.5 # never abandon primary goal in favor of secondary goals
        goaldiff_weighted = self.ep_goalweight * (achieved_obs - desired_obs)
        goaldist = np.linalg.norm(goaldiff_weighted, axis=-1)

        self.ep_goaldists.append(goaldist)

        goalconv = 0
        is_converging = 0
        if len(self.ep_goaldists) > 1:
            goalconv = self.ep_goaldists[-2] - self.ep_goaldists[-1]
            is_converging = np.sign(goalconv) / 2 # normalized to [-0.5,0.5]

        self.ep_goalconvs.append(goalconv)


        # ========= META OBS

        if self.cfg.MetaObservation.IS_ENABLED:
            metaobs = np.array([])

            goalweight_hash = vector_to_uniform_scalar(self.ep_goalweight, len(self.ep_goalweight))
            metaobs = np.append(metaobs, goalweight_hash)
            metaobs = np.append(metaobs, goaldist)
            metaobs = np.append(metaobs, desired_obs.ravel())
            # record_dist = max(0, goaldist - self.tr_goaldist_min)
            # metaobs.append(record_dist) # may hinder retraining of restored policy (record-reset)
            metaobs = np.append(metaobs, goalconv)
            metaobs = np.append(metaobs, is_converging)
            obs = np.append(obs, metaobs)


        # metagoal(s) only
        # achieved_goal = 0.5 * goaldist + 0.5 * goalconv
        achieved_goal = goalconv
        # desired_goal = self.ep_reward_threshold
        desired_goal = 1.0  # or arbitrary big number for max.??

        dictobs = dict(
                observation=obs,
                achieved_goal=np.array([achieved_goal]),
                desired_goal=np.array([desired_goal]),
            )

        # if self.is_plot and self.ep_num_steps % cfg.General.STEPSKIP_PLOT == 0:
        #     achieved_img_annotated = draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
        #     desired_img_annotated = draw_landmarks_on_image(desired_img.numpy_view(), desired_pose)
        #     if self.parallel_plot_queue.empty():
        #         self.parallel_plot_queue.put_nowait((achieved_img_annotated, desired_img_annotated, achieved_pose, desired_pose))

        return dictobs


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        # return np.clip(achieved_goal, 0, None)
        return achieved_goal


    def reset_model(self):
        obs_init = None

        if self.ep_current_obs and len(self.ep_goaldists) > 0:
            # episode report
            LOG.debug('ep_lives %s', self.ep_lives)
            LOG.debug('ep_num_steps %s', self.ep_num_steps)
            LOG.debug('ep_num_steps_goal_zone %s', self.ep_num_steps_goal_zone)
            LOG.debug('ep_first_reward_step %s', self.ep_first_reward_step)
            LOG.debug('ep_goaldims_active %s', np.nonzero(self.ep_goalweight)[0])
            LOG.debug('ep_goaldist_desired %s', self.ep_current_obs['desired_goal'])
            LOG.debug('ep_goaldist_first %s', self.ep_goaldists[0])
            LOG.debug('ep_goaldist_min %s', np.min(self.ep_goaldists))
            LOG.debug('ep_goaldist_mean %s', np.mean(self.ep_goaldists))
            LOG.debug('ep_goaldist_max %s', np.max(self.ep_goaldists))
            LOG.debug('ep_goaldist_last %s', self.ep_current_obs['achieved_goal'])
            LOG.debug('ep_reward_threshold %s %s', cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT, self.ep_reward_threshold)
            LOG.debug('ep_traj_is_halved %s', self.ep_traj_is_halved)
            LOG.debug('ep_rewards_mean %s', self.ep_rewards_mean)
            LOG.debug('ep_goalzone_per_step %s', np.round(self.ep_num_steps_goal_zone /  self.ep_num_steps, 2))
            LOG.debug('ep_goalconv_mean %s', np.mean(np.diff(self.ep_goaldists)))
            LOG.debug('\n')

        # if self.ep_num_steps > self.cfg.PracticeSpace.STEPS_INVINCIBLE_SPAWN:
        #     if self.cfg.TrajectoryHalving.IS_ENABLED and not self.is_eval:
        #         # if self.ep_goaldist_min < self.last_ep_goaldist_min:
        #         if self.ep_lives > 0:
        #             LOG.info('LAST SAVEPOINT.') # noisy?
        #             obs_init = self._reset_half_episode()

        #             # correcting ep init state
        #             if(obs_init['achieved_goal'] > self.ep_goaldist_min):
        #                 drift = self.ep_goaldist_min - obs_init['achieved_goal']
        #                 LOG.debug('SAVEPOINT STATE HAS DRIFTED OFF-GRID (NOISE?). correcting... %s %s', self.fep_savepoint_steps, drift)
        #                 obs_init['achieved_goal'] = self.ep_goaldist_min
        #                 if np.abs(drift) > self.ep_reward_threshold:
        #                     LOG.warning('DRIFT IS GREATER THAN THRESHOLD! CONSIDER REDUCE NOISE OR INCREASE THRESHOLD.')

        if not obs_init:
            obs_init = self._reset_full_episode()
            LOG.info('\nNEW GAME.')

        # noisy goaldist detection (savepoint may not same/best anymore)
        self.ep_goaldist_min = obs_init['achieved_goal']
        self.ep_goaldist_max = obs_init['achieved_goal']

        if not self.is_eval and self.tr_num_steps > 10:
            self.outfile_ep_rewards_mean.write('%s\n' % (self.ep_rewards_mean))
            self.outfile_ep_rewards_mean.flush()

        self._reset()
        self.ep_states.append((self.data.qpos.flat.copy(), self.data.qvel.flat.copy()))
        LOG.debug('fep_savepoint_steps %s', self.fep_savepoint_steps)
        LOG.debug('fep_num_steps_goal_zone %s', self.fep_savepoint_steps_goal_zone + self.ep_num_steps_goal_zone)
        LOG.debug('fep_savepoint_goaldist %s', obs_init['achieved_goal'])
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min)
        LOG.debug('fep_goaldist_max %s', self.fep_goaldist_max)

        return obs_init
    

    def _reset_full_episode(self):
        obs_init = super().reset_model()
        self.fep_savepoint_steps = 0
        self.fep_savepoint_steps_goal_zone = 0
        self.fep_goaldist_init = obs_init['achieved_goal']
        self.fep_goaldist_min = obs_init['achieved_goal']
        self.fep_obs_init = obs_init
        self.fep_goaldims_primary = np.random.randint(4, size=1) # multiple?
        # TODO if random, then only secondary interval
        self.fep_goaldims_secondary = np.random.randint(len(self.ep_goalweight), size=1) # multiple?
        # self.desired_obs = self._get_desired_obs()
        self.last_ep_goaldist_min = np.inf
        self.last_ep_rewards_mean = 0
        self.ep_traj_is_halved = False
        self.ep_lives = cfg.General.MAX_LIVES
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
        if self.fep_rewards_sum < 0:
            self.tr_feps_consecutive_neg += 1
        else:
            self.tr_feps_consecutive_neg = 0

        LOG.debug('tr_feps_total %s', self.tr_feps_total)
        LOG.debug('tr_feps_consecutive_neg %s', self.tr_feps_consecutive_neg)
        LOG.debug('tr_obsdims %s', obs_init['observation'].shape[-1])
        LOG.debug('tr_obs_min %s', np.min(obs_init['observation']))
        LOG.debug('tr_obs_mean %s', np.mean(obs_init['observation']))
        LOG.debug('tr_obs_max %s', np.max(obs_init['observation']))
        LOG.debug('tr_goaldist_min %s', self.tr_goaldist_min)
        LOG.debug('tr_goaldist_max %s', self.tr_goaldist_max)
        LOG.debug('tr_goaldist_mins_mean %s', self.tr_goaldist_mins_mean)
        LOG.debug('tr_goaldist_maxs_mean %s', self.tr_goaldist_maxs_mean)
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min)
        LOG.debug('fep_goaldist_max %s', self.fep_goaldist_max)
        LOG.debug('fep_rewards_sum %s', self.fep_rewards_sum)

        self.fep_rewards_sum = 0
        return obs_init


    def _reset_half_episode(self):
        idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
        self.fep_savepoint_steps += idx_halving
        if idx_halving > 0: # improved
            self.fep_savepoint_steps_goal_zone += self.ep_num_steps_goal_zone

        LOG.info('halving at: %s %s', idx_halving, len(self.ep_states))
        qpos, qvel = self.ep_states[idx_halving]
        # TODO remove or add noise?
        # qpos, qvel = self._add_noise(qpos, qvel)
        self.set_state(qpos, qvel)
        self.ep_states.append((qpos, qvel))
        self.ep_traj_is_halved = True
        self.last_ep_rewards_mean = self.ep_rewards_mean
        self.last_ep_goaldist_min = self.ep_goaldist_min
        
        self.ep_states = []
        self.ep_dictobs = []
        self.ep_goaldists = []
        self.ep_goalconvs = []

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
        self.ep_states = []
        self.ep_num_steps_goal_zone = 0
        self.ep_count_fails_pose_detection = 0

        if not self.landmarker_achieved:
            self.landmarker_achieved = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_achieved)
            self.landmarker_desired = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_desired)


    def _get_idx_for_trajectory_halving(self, strat):
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
                idx_step = max(0, self.ep_last_reward_step)
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


    def _normalize_to_limits(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)



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
