
import numpy as np
import time
import logging
import git
from types import SimpleNamespace
from . import pose_imitation_cfg as cfg
from gymnasium import spaces

import multiprocessing
from multiprocessing.queues import Empty
import matplotlib.pyplot as plt
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D

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

OBS_NORMALIZE_Z_SCORE = False

# 40k, noPen:   slow turn, little standing
# 100k, noPen:  worsens again significantly (bend legs, early termination)
# 40k, correctHeight:   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f4559e6/le-pose-imitation-v4/23-19-34/rl_model_finished
# 100k, correctHeight: no change, still worsens with training time /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f4559e6/le-pose-imitation-v4/23-19-34_restored/rl_model_finished
class PoseImitationEnv(HumanoidEnv):


    def __init__(self, is_eval=False, is_plot=True, log_level=logging.INFO):
        LOG.setLevel(log_level)

        HumanoidEnv.__init__(self,
                             exclude_current_positions_from_observation=True,
                             width=cfg.General.RENDER_IMAGE_SIZE,
                             height=cfg.General.RENDER_IMAGE_SIZE,
                             xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/humanoid_face.xml')
                            #  xml_file=PATH_GIT_WORKING_DIR + '/src/custom_envs/le_humanoid_pose/robotis_op3/scene.xml')
        self.frame_skip: 5 = cfg.General.FRAMESKIP_STEP

        assert cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT and cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT, 'cannot plot in a step with no pose render (and detection'

        self.cfg = cfg
        self.is_plot = is_plot
        self.is_eval = is_eval
        self.outfile_ep_rewards_mean = open('ep_rewards_mean.dat', 'w')

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
        # obspace_total_dims += self.observation_space.shape[0] # super
        # obspace_total_dims += 2 # falling, height
        obspace_total_dims += cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION
        
        if self.cfg.MetaObservation.IS_ENABLED:
            # obspace_total_dims += 2 # falling, height
            obspace_total_dims += 3 # goaldist, goal_convergence, is_converging
            obspace_total_dims += cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION # desired etc.

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
        self.init_qpos[6] = -1.4 # face towards camera
        self.tr_feps_total = 0
        self.tr_feps_consecutive_neg = 0
        self.tr_total_zero_sum_eps = 0
        self.tr_total_zero_sum_eps_steps = 0
        self.tr_goaldist_personal_best = np.inf
        self.tr_num_steps: int = 0
        self.fep_savepoint_steps = 0
        self.fep_goaldist_init = np.inf
        self.fep_rewards_sum = -1
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goaldist_min: float = np.inf
        self.ep_num_steps: int = 0
        self.ep_goaldist_min: float = np.inf
        self.ep_goaldist_max: float = -np.inf
        self.ep_goaldim_active = -1
        self.ep_goalweight = -1
        self.ep_lives = cfg.General.MAX_LIVES
        # constant threshold
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT

        self.landmarker_achieved = None
        self.landmarker_desired = None
        self.last_ob_pose_achieved = np.full(cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION, 1)
        self.last_ob_pose_desired = np.full(cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION, 1)

        if self.is_plot:
            self.parallel_plot_queue = multiprocessing.Queue()
            multiprocessing.log_to_stderr(logging.DEBUG)
            multiprocessing.Process(target=parallel_plot, args=((self.parallel_plot_queue,)), daemon=True).start()

        self._reset()
        LOG.debug('le-walker-2d initialized.')
        LOG.debug('observation_space %s', observation_space)
        LOG.debug('goal_space %s', goal_space)


    def step(self, action):
        # self.frame_skip = random.randint(0, 100)
        self.do_simulation(action, self.frame_skip)

        info = {}
        info['success'] = False
        obs = self._get_obs()
        self.ep_obs_cur = obs

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.ep_states.append((qpos, qvel))

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        if reward:
            self.ep_num_steps_goal_zone += 1

        # NUDGING (healing health)
        if obs['achieved_goal'] < self.ep_goaldist_min:
            LOG.debug(f"IMPROVED: {obs['achieved_goal']} < {self.ep_goaldist_min}")
            self.ep_goaldist_min = obs['achieved_goal']
            self.ep_lives = cfg.General.MAX_LIVES
            reward = 1
        # DENUDGING
        # save-and-go ("souls-like emulator")
        # 40k:  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22/rl_model_finished
        # 100k: /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored/rl_model_finished
        # 200k: natural+robust stabilization? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored_restored/rl_model_finished
        # 400k: /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored_restored_restored/rl_model_finished
        # 1M:   a little bit slower than without SAG (due less same init state (full resets)? -> adjust num lives?) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored_restored_restored_restored/rl_model_finished
        # 1.5M, comb-through, off-grid, lives100, goalpos:   slow /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored_restored_restored_restored_restored/rl_model_finished
        # 1.5M(!), comb-through, on-grid, lives100, goalpos:    better /home/t14/Documents/tuhh/dsf/Scilab-RL/data/ee8db7f/le-pose-imitation-v4/23-20-22_restored_restored_restored_restored_restored/rl_model_finished
        #                                                    stands faster than goaldir /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/07-11-30/rl_model_finished
        # 3M                                            :   is it slightly better? or not progressing further? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/07-11-30_restored/rl_model_finished
        # 1.0M, comb-through, on-grid, lives100, goaldir, z-standard:   not working at all (lossfunc degen.) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/12-24-30/rl_model_finished
        # 0.5M, comb-through, on-grid, lives100, goaldir:   fast resemblence /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/15-50-09/rl_model_finished
        # 1.0M,                                         :   clearly attempting /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/15-50-09_restored/rl_model_finished
        # 2M,                                           :   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/15-50-09_restored_restored/rl_model_finished
        # 3M,                                           :   progress, but slower than goalpos (but maybe more general?) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/15-50-09_restored_restored_restored/rl_model_finished
        # 1M, comb-through, on-grid, lives10, goalpos:  :   slower on arms moving /home/t14/Documents/tuhh/dsf/Scilab-RL/data/39e45d1/le-pose-imitation-v4/12-48-33/rl_model_finished
        # 3M,                                      , noDenudge:

        # elif obs['achieved_goal'] > self.ep_goaldist_max:
        #     LOG.debug(f"DETERIORATE: {obs['achieved_goal']} > {self.ep_goaldist_max}")
        #     self.ep_goaldist_max = obs['achieved_goal']
        #     reward = -1

        if reward:
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        terminated = False
        truncated = False

        # space constraint
        if self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE:
            height = super()._get_obs()[0]

            if obs['achieved_goal'] > self.fep_goaldist_init * 1.1:
                LOG.info('GOAL TOO FAR AWAY.')
                terminated = True
                self.ep_lives -= 1

            if self.ep_count_fails_pose_detection > 10:
                LOG.info('TOO MANY DETECTION FAILURES.')
                terminated = True
                self.ep_lives -= 1

            
            # if height < 0.5 or height > 2.0:
            #     # TODO learn default standing-pose first?
            #     LOG.debug('OUTSIDE: FELL DOWN! %s', height)
            #     terminated = True
            #     # dont neutralize already pos. eps.?
            #     # if not self.ep_rewards_mean and not reward:
            #     reward = self.cfg.PracticeSpace.REWARD_ON_TERMINATE
            #     self.ep_lives -= 1

        self.fep_rewards_sum += reward
        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            LOG.debug('TRUNCATED.')
            info['success'] = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            truncated = True

        result = obs, reward, terminated, truncated, info
        return result


    # obs = superobs(phys.) + achieved_obs(vis.) + metaobs
    def _get_obs(self):
        obs = []
        superobs = super()._get_obs()
        # obs.extend(superobs)

        achieved_pose = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        desired_pose = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

        # detect pose only every nth step, else use last valid
        if self.ep_num_steps % cfg.General.STEPSKIP_DETECT == 0:
            # renders only rgb (cant render multiple modes simultanously)
            render_tmp = self.render_mode
            self.render_mode = 'rgb_array'

            # https://github.com/jurgisp/memory-maze/issues/26
            achieved_img = self.render().copy() # MUJOCO_GL=glfw
            self.render_mode = render_tmp
            achieved_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=achieved_img)
            # TODO get desired img from video?
            desired_img = self.desired_img

            # bottleneck start
            # t = time.perf_counter()
            # https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker#detect_for_video
            video_timestamp_ms = int(time.process_time_ns() / 1000 + self.ep_num_steps)
            achieved_pose = self.landmarker_achieved.detect_for_video(achieved_img, video_timestamp_ms)
            if not self.desired_pose:
                # only once at the beginning (still image)
                self.desired_pose = self.landmarker_desired.detect(desired_img)
            desired_pose = self.desired_pose
            # LOG.debug(time.perf_counter() - t)
            # bottleneck end

        elif self.render_mode == 'human':
            self.render()


        # get normalized goal distance
        achieved_obs = np.array([])

        # achieved_ob_fall = self._normalize_to_limits(superobs[24], 0, -2.5)
        # achieved_obs = np.append(achieved_obs, achieved_ob_fall)

        # achieved_ob_height = superobs[0]
        # achieved_ob_height = self._normalize_to_limits(achieved_ob_height, 1.0, 2.0)
        # achieved_obs = np.append(achieved_obs, achieved_ob_height)

        # achieved_ob_pose = []
        achieved_ob_pose = self.last_ob_pose_achieved
        if achieved_pose.pose_world_landmarks:
            # only first detected pose
            # height-dependent vs. -independent
            # achieved_ob_pose = [(landmark.x, landmark.y - superobs[0], landmark.z) for landmark in achieved_pose.pose_world_landmarks[0]]
            achieved_ob_pose = [(landmark.x, landmark.y, landmark.z) for landmark in achieved_pose.pose_world_landmarks[0]]
            achieved_ob_pose = np.array(achieved_ob_pose)[cfg.General.LANDMARK_GROUPS_FLAT]
            achieved_ob_pose = self._normalize_to_limits(achieved_ob_pose, -1, 1)
            self.last_ob_pose_achieved = achieved_ob_pose
            self.ep_count_fails_pose_detection = 0
        else:
            self.ep_count_fails_pose_detection += 1
            
        achieved_obs = np.append(achieved_obs, achieved_ob_pose)
        obs.extend(achieved_obs)


        desired_obs = np.array([])

        # desired_ob_fall = 0
        # desired_obs = np.append(desired_obs, desired_ob_fall)

        # desired_ob_height = 1.3
        # desired_ob_height = self._normalize_to_limits(desired_ob_height, 1.0, 2.0)
        # desired_obs = np.append(desired_obs, desired_ob_height)

        # desired_ob_pose = []
        desired_ob_pose = self.last_ob_pose_desired
        if desired_pose.pose_world_landmarks:
            # only first detected pose
            # desired_ob_pose = [(landmark.x, landmark.y - 1.2, landmark.z) for landmark in desired_pose.pose_world_landmarks[0]]
            desired_ob_pose = [(landmark.x, landmark.y, landmark.z) for landmark in desired_pose.pose_world_landmarks[0]]
            desired_ob_pose = np.array(desired_ob_pose)[cfg.General.LANDMARK_GROUPS_FLAT]
            desired_ob_pose = self._normalize_to_limits(desired_ob_pose, -1, 1)
            self.last_ob_pose_desired = desired_ob_pose
        desired_obs = np.append(desired_obs, desired_ob_pose)


        # conseq. one-timed single subdim. ("comb-through" training), bigger stepskips
        # very different goaldists per ep.!
        # 40k, th0.1:  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/23-58-04/rl_model_finished
        # 100k, th0.1:     tumbles /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/23-58-04_restored/rl_model_finished
        # 40k, th0.05:  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/00-23-10/rl_model_finished
        # 100k:         /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/00-23-10_restored/rl_model_finished
        # 200k:         /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/00-23-10_restored_restored/rl_model_finished
        # 400k:         left arm moves up? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/00-23-10_restored_restored_restored/rl_model_finished
        # 2.4M(!!):     improvement! attempting resemblence, still too much falling? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/00-23-10_restored_restored_restored_restored/rl_model_finished
        # 40k, base-dims, ppo:  more exploration, more dynamic, less careful, more curious/eager to learn?  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/8bc7ad5/le-pose-imitation-v4/19-59-06/rl_model_finished
        # 40k                :  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/11-44-16/rl_model_finished  
        # 100k               :  more eager, resemblence, still unstable /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/11-44-16_restored/rl_model_finished
        # 200k               :  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/11-44-16_restored_restored/rl_model_finished
        # 1M                 :  no apparent progress/convergence /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/11-44-16_restored_restored_restored/rl_model_finished
        # 2M                 :  still no progress /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/11-44-16_restored_restored_restored_restored/rl_model_finished
        # 100k, cleanParams  :  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f5b5c5f/le-pose-imitation-v4/23-43-30/rl_model_finished
        # 1M                 :  still no progress /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f5b5c5f/le-pose-imitation-v4/23-43-30_restored/rl_model_finished
        # 200k, zooParams    :  no real progress /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f5b5c5f/le-pose-imitation-v4/01-14-45/rl_model_finished
        # 1M                 :  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f5b5c5f/le-pose-imitation-v4/01-14-45_restored/rl_model_finished
        # maybe even more? (~5h sac)

        # 40k, base-dims, (sbx.)sac :   /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34/rl_model_finished
        # 100k                      :   attempting, resemblence, stabilising? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34_restored/rl_model_finished
        # 200k                      :   stronger attempts, both arms wildly moving /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34_restored_restored/rl_model_finished
        # 1M(!!!)                   :   truncation, stable, closest resemblence without falling (no feet!) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34_restored_restored_restored/rl_model_finished
        # 1.2M, pose2               :   collapsing again (legs?), attempting arm resemblance /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34_restored_restored_restored_restored/rl_model_finished
        # 2M                        :   slow, but noticeable progress (difficult humanoid?) /home/t14/Documents/tuhh/dsf/Scilab-RL/data/15372b1/le-pose-imitation-v4/10-49-34_restored_restored_restored_restored_restored/rl_model_finished

        # 100k, arms-only :    simplified, arms resemblance, avoid falling by jumping? /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/14-19-48_restored/rl_model_finished
        # 200k, arms-only, frameskip10 :    /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/15-28-00/rl_model_finished
        self.ep_goalweight = np.zeros(desired_obs.shape)
        # self.ep_goalweight[0] = 1 # base dim
        # self.ep_goalweight[1] = 1 # base dim
        # comb-through (only after stable/nonterminating? truncation + success cond.)
        self.ep_goalweight[self.ep_goaldim_active] = 1

        # only base-dims (stabilize, vs termination), pen
        # 40k, pen      /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/17-28-52/rl_model_finished
        # 100k          /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/17-28-52_restored/rl_model_finished
        # 100k(!), no pen:  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/17-54-08/rl_model_finished
        # 300k           :  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/17-54-08_restored/rl_model_finished
        # 1M             :  slightly walking to not fall, truncation /home/t14/Documents/tuhh/dsf/Scilab-RL/data/b799fe5/le-pose-imitation-v4/17-54-08_restored_restored/rl_model_finished
        goaldiff_weighted = self.ep_goalweight * (achieved_obs - desired_obs)
        goaldist = np.linalg.norm(goaldiff_weighted, axis=-1)
        self.ep_goaldists.append(goaldist)

        if self.cfg.MetaObservation.IS_ENABLED:
            metaobs = []

            metaobs.append(goaldist)
            # TODO test/dev in specialized env. (eg. reach-env.)
            # meta-observe normalized direction instead of pose coords (more general?)
            # TODO or both?
            metaobs.extend(desired_obs)
            # goaldirection = goaldiff_weighted / np.linalg.norm(goaldiff_weighted)
            # metaobs.extend(goaldirection)

            if len(self.ep_goaldists) > 1:
                goal_convergence = self.ep_goaldists[-2] - self.ep_goaldists[-1]
                metaobs.append(goal_convergence)
                is_converging = np.sign(goal_convergence)
                metaobs.append(is_converging)
            else:
                metaobs.extend([0,0])
            obs.extend(metaobs)

        obs = np.array(obs)
        goaldist = np.array(goaldist)

        dictobs = dict(
                observation=obs,
                achieved_goal=goaldist,
                desired_goal=self.ep_reward_threshold,
            )

        if self.is_plot and self.ep_num_steps % cfg.General.STEPSKIP_PLOT == 0:
            achieved_img_annotated = draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
            desired_img_annotated = draw_landmarks_on_image(desired_img.numpy_view(), desired_pose)
            if self.parallel_plot_queue.empty:
                self.parallel_plot_queue.put((achieved_img_annotated, desired_img_annotated, achieved_pose, desired_pose))

        return dictobs


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        reward = (achieved_goal < desired_goal)
        return reward.astype(np.float64)


    def reset_model(self):
        obs_init = None

        if self.ep_num_steps > 0:
            # episode report
            LOG.debug('ep_lives %s', self.ep_lives)
            LOG.debug('ep_num_steps %s', self.ep_num_steps)
            LOG.debug('ep_num_steps_goal_zone %s', self.ep_num_steps_goal_zone)
            LOG.debug('ep_first_reward_step %s', self.ep_first_reward_step)
            LOG.debug('ep_goaldim_active %s %s', self.ep_goaldim_active, np.nonzero(self.ep_goalweight)[0])
            LOG.debug('ep_goaldist_desired %s', self.ep_obs_cur['desired_goal'])
            LOG.debug('ep_goaldist_first %s', self.ep_goaldists[0])
            LOG.debug('ep_goaldist_min %s', min(self.ep_goaldists))
            LOG.debug('ep_goaldist_mean %s', np.mean(self.ep_goaldists))
            LOG.debug('ep_goaldist_max %s', max(self.ep_goaldists))
            LOG.debug('ep_goaldist_last %s', self.ep_obs_cur['achieved_goal'])
            LOG.debug('ep_reward_threshold %s %s', cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT, self.ep_reward_threshold)
            LOG.debug('ep_traj_is_halved %s', self.ep_traj_is_halved)
            LOG.debug('ep_rewards_mean %s', self.ep_rewards_mean)
            if self.ep_rewards_mean == 0:
                LOG.debug('WARNING: zero-sum-ep. => wasted ep.?')
                self.tr_total_zero_sum_eps += 1
                self.tr_total_zero_sum_eps_steps += self.ep_num_steps
            LOG.debug('\n')
            if self.is_eval:
                self.outfile_ep_rewards_mean.write(f"{self.ep_rewards_mean}\n")
                self.outfile_ep_rewards_mean.flush()
        
        if self.ep_num_steps > 1:
            if self.cfg.TrajectoryHalving.IS_ENABLED and not self.is_eval:
                # if self.ep_goaldist_min < self.last_ep_goaldist_min:
                if self.ep_lives > 0:
                    LOG.debug('LAST SAVEPOINT.') # noisy?
                    obs_init = self._reset_half_episode()
                    if(obs_init['achieved_goal'] > self.ep_goaldist_min):
                        LOG.debug('SAVEPOINT HAS DRIFTED OFF-GRID (NOISE?). correcting... %s %s', self.fep_savepoint_steps, self.ep_goaldist_min - obs_init['achieved_goal'])
                        # obs_init = None
                        obs_init['achieved_goal'] = self.ep_goaldist_min

        if not obs_init:
            LOG.debug('NEW GAME.')
            obs_init = self._reset_full_episode()

        # WIP half (more nudging, minigame) vs. full (once nudging, orig.game)
        # noisy goaldist detection (savepoint may not same/best anymore)
        self.ep_goaldist_min = obs_init['achieved_goal']
        self.ep_goaldist_max = obs_init['achieved_goal']

        self._reset()
        self.ep_goaldists.append(obs_init['achieved_goal'])
        self.ep_states.append((self.data.qpos.flat.copy(), self.data.qvel.flat.copy()))
        self.fep_goaldist_min = min(self.fep_goaldist_min, self.ep_goaldist_min)
        LOG.debug('fep_savepoint_steps %s', self.fep_savepoint_steps)
        LOG.debug('fep_savepoint_goaldist %s', obs_init['achieved_goal'])
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_goaldist_min %s', self.fep_goaldist_min) # may be noisy and not (easily) repeatable

        return obs_init
    

    def _reset_full_episode(self):
        obs_init = super().reset_model()
        self.tr_goaldist_personal_best = min(self.tr_goaldist_personal_best, self.ep_goaldist_min)
        self.fep_savepoint_steps = 0
        self.fep_goaldist_init = obs_init['achieved_goal']
        self.fep_goaldist_min = obs_init['achieved_goal']
        # self.desired_obs = self._get_desired_obs()
        self.last_ep_goaldist_min = np.inf
        self.last_ep_rewards_mean = 0
        self.ep_traj_is_halved = False
        self.ep_goaldim_active = (self.ep_goaldim_active + 1) % len(self.ep_goalweight)
        self.ep_lives = cfg.General.MAX_LIVES
        # TODO redo noise?
        # noisy relative threshold (varies by initial state noise)
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * obs_init['achieved_goal']
        # self.ep_goaldist_min = obs_init['achieved_goal']
        # self.ep_goaldist_max = obs_init['achieved_goal']

        self.tr_feps_total += 1
        if self.fep_rewards_sum < 0:
            self.tr_feps_consecutive_neg += 1
        else:
            self.tr_feps_consecutive_neg = 0

        LOG.debug('tr_feps_total %s', self.tr_feps_total)
        LOG.debug('tr_feps_consecutive_neg %s', self.tr_feps_consecutive_neg)
        LOG.debug('tr_total_zero_sum_eps %s', self.tr_total_zero_sum_eps)
        LOG.debug('tr_total_zero_sum_eps_steps %s', self.tr_total_zero_sum_eps_steps)
        LOG.debug('tr_count_goal_reached %s', self.ep_num_steps_goal_zone)
        LOG.debug('tr_goaldist_personal_best %s', self.tr_goaldist_personal_best)
        LOG.debug('fep_goaldist_init %s', self.fep_goaldist_init)
        LOG.debug('fep_rewards_sum %s', self.fep_rewards_sum)
        self.fep_rewards_sum = 0
        return obs_init


    def _reset_half_episode(self):
        idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
        self.fep_savepoint_steps += idx_halving
        qpos, qvel = self.ep_states[idx_halving]
        # qpos, qvel = self._add_noise(qpos, qvel)
        self.set_state(qpos, qvel)
        self.ep_traj_is_halved = True
        self.last_ep_rewards_mean = self.ep_rewards_mean
        self.last_ep_goaldist_min = self.ep_goaldist_min

        obs_init = self._get_obs()
        return obs_init


    def _reset(self):
        self.ep_rewards_mean: float = -1
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_obs_cur = None
        self.ep_goaldists = []
        self.ep_states = []
        self.ep_is_perfect = False
        self.ep_num_steps_goal_zone = 0
        # LOG.debug('desired_obs', self.desired_obs)

        # landmarker reset: better/correct detection of start pose
        # TODO wait for efficient reset() implementation in API
        # workaround by re-create
        # if self.landmarker_achieved:
        #     self.landmarker_achieved.close()
        #     del self.landmarker_achieved
        #     self.landmarker_achieved = None
        # if self.landmarker_desired:
        #     self.landmarker_desired.close()
        #     del self.landmarker_desired
        #     self.landmarker_desired = None

        if not self.landmarker_achieved:
            self.landmarker_achieved = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_achieved)
            self.landmarker_desired = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_desired)


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
        # achieved_img, desired_img, achieved_pose, desired_pose = 0
        # while not queue.empty(): # get only latest
        achieved_img, desired_img, achieved_pose, desired_pose = queue.get()
    
        plot_desired.set_data(desired_img)
        plot_desired.draw(plot_desired.get_figure().canvas.get_renderer())

        plot_achieved.set_data(achieved_img)
        plot_achieved.draw(plot_achieved.get_figure().canvas.get_renderer())

        # plot topology connections
        # https://github.com/stebusse/mediapipe-plot-pose-live/blob/main/plot_pose_live.py
        extplot.clear()
        extplot.set_xlim3d(-1, 1)
        extplot.set_ylim3d(-1, 1)
        extplot.set_zlim3d(1, -1) # flip z-axis

        if achieved_pose.pose_world_landmarks:
            for group in cfg.General.LANDMARK_GROUPS:
                plotX = [achieved_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [achieved_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [achieved_pose.pose_world_landmarks[0][i].z for i in group]
                if 11 in group: # right side
                    extplot.plot(plotX, plotZ, plotY, color='red')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='red', linestyle = 'dashed')

        if desired_pose.pose_world_landmarks:
            for group in cfg.General.LANDMARK_GROUPS:
                plotX = [desired_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [desired_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [desired_pose.pose_world_landmarks[0][i].z for i in group]
                if 11 in group: # right side
                    extplot.plot(plotX, plotZ, plotY, color='green')
                else:
                    extplot.plot(plotX, plotZ, plotY, color='green', linestyle = 'dashed')
        
        
        extplot.draw(extplot.get_figure().canvas.get_renderer())
        plt.pause(0.00001)
