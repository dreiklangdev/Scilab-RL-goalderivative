
import numpy as np
import time
import logging
import git
from types import SimpleNamespace
from . import pose_imitation_cfg as cfg
from gymnasium import spaces

import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D

from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker

PATH_GIT_WORKING_DIR = git.Repo('.', search_parent_directories=True).working_tree_dir

LANDMARK_GROUPS = [
    [8, 6, 5, 4, 0, 1, 2, 3, 7],   # eyes
    [10, 9],                       # mouth
    [11, 13, 15, 17, 19, 15, 21],  # right arm
    [11, 23, 25, 27, 29, 31, 27],  # right body side
    [12, 14, 16, 18, 20, 16, 22],  # left arm
    [12, 24, 26, 28, 30, 32, 28],  # left body side
    [11, 12],                      # shoulder
    [23, 24],                      # waist
]

# https://chuoling.github.io/mediapipe/solutions/pose.html
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
# https://saiwa.ai/blog/openpose-vs-mediapipe/
# https://jetson-docs.com/libraries/mediapipe/overview

# https://pytorch.org/rl/0.6/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
# https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb

# https://github.com/google-ai-edge/mediapipe/issues/5325
# https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
# https://github.com/google-ai-edge/mediapipe/issues/5325
# TODO reduce goal features?
# TODO terminate on missing pose detection?


class PoseImitationEnv(HumanoidEnv):


    def __init__(self, is_plot=True):

        HumanoidEnv.__init__(self, exclude_current_positions_from_observation=True, width=cfg.General.RENDER_IMAGE_SIZE, height=cfg.General.RENDER_IMAGE_SIZE)
        self.frame_skip: 5 = cfg.General.FRAMESKIP_STEP

        assert cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT and cfg.General.STEPSKIP_PLOT >= cfg.General.STEPSKIP_DETECT, 'cannot plot in a step with no pose render (and detection'

        self.cfg = cfg
        self.is_plot = is_plot
        img_array = image.imread(PATH_GIT_WORKING_DIR + '/mediapipe/poses/pose1.jpg')
        self.desired_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array.copy())

        # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
        self.landmarker_options_achieved = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=PATH_GIT_WORKING_DIR + '/mediapipe/model/pose_landmarker_lite.task',            
                # cpu vs gpu
                # https://forums.developer.nvidia.com/t/how-to-install-opengl-libs-of-nvidia/175409
                # https://stackoverflow.com/questions/77707532/how-to-check-for-and-enforce-gpu-usage-for-mediapipe-frame-processing/79202595#79202595
                # prime-select nvidia
                # glxinfo | grep -i opengl
                # MUJOCO_GL=egl|glfw|osmesa %python ...% (glfw seems fastest)
                delegate=BaseOptions.Delegate.GPU),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1)
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
        obspace_total_dims += cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION + 1 # achieved

        if self.cfg.MetaObservation.IS_ENABLED:
            obspace_total_dims += cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION + 4 # desired etc.

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
        self.tr_feps_total = 0
        self.tr_feps_consecutive_neg = 0
        self.tr_total_zero_sum_eps = 0
        self.tr_total_zero_sum_eps_steps = 0
        self.tr_goaldist_personal_best: float = np.inf
        self.fep_rewards_sum = -1
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goaldist_min: float = np.inf
        self.ep_num_steps: int = 0
        # constant threshold
        self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * 3
        
        self.landmarker_achieved = None
        self.landmarker_desired = None
        self.last_detected_obs_achieved = np.full(cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION, 1)
        self.last_detected_obs_desired = np.full(cfg.General.OBSERVATION_DIMS_VISUAL_DETECTION, 1)

        if self.is_plot:
            self.parallel_plot_queue = multiprocessing.Queue()
            multiprocessing.log_to_stderr(logging.DEBUG)
            multiprocessing.Process(target=parallel_plot, args=((self.parallel_plot_queue,)), daemon=True).start()

        self._reset()
        print('le-walker-2d initialized.')
        print('observation_space', observation_space)
        print('goal_space', goal_space)


    def step(self, action):
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
            if self.ep_first_reward_step < 0:
                self.ep_first_reward_step = self.ep_num_steps

        terminated = False
        truncated = False

        # space constraint
        if self.cfg.PracticeSpace.IS_TERMINATE_ON_OUTSIDE_PRACTICE_SPACE:
            height = obs['observation'][0]
            if height < 1.0 or height > 2.0:
                # TODO learn default standing-pose first?
                print('OUTSIDE: FELL DOWN!', height)
                terminated = True
                # dont neutralize already pos. eps.
                if not self.ep_rewards_mean and not reward:
                    reward = self.cfg.PracticeSpace.REWARD_ON_TERMINATE

        # time constraint
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
                        reward = self.cfg.PracticeSpace.REWARD_ON_TERMINATE

        self.fep_rewards_sum += reward
        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        if self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            print('TRUNCATED.')
            info['success'] = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            truncated = True

        result = obs, reward, terminated, truncated, info
        return result


    # obs = superobs(phys.) + achieved_obs(vis.) + metaobs
    # 40k, unnormalized:    slow, tries standing, no turning /home/t14/Documents/tuhh/dsf/Scilab-RL/data/28af96a/le-pose-imitation-v4/01-22-59/rl_model_finished
    def _get_obs(self):
        obs = []
        superobs = super()._get_obs()
        obs.extend(superobs)

        # detect pose only every nth step, else use last valid
        if self.ep_num_steps % cfg.General.STEPSKIP_DETECT == 0:
            # renders only rgb (cant render multiple modes simultanously)
            self.render_mode = 'rgb_array'

            # https://github.com/jurgisp/memory-maze/issues/26
            achieved_img = self.render().copy() # MUJOCO_GL=glfw
            achieved_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=achieved_img)
            # TODO get desired img from video?

            # bottleneck start
            # t = time.perf_counter()
            # https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker#detect_for_video
            video_timestamp_ms = int(time.process_time_ns() / 1000 + self.ep_num_steps)
            achieved_pose = self.landmarker_achieved.detect_for_video(achieved_img, video_timestamp_ms)
            desired_pose = self.landmarker_desired.detect(self.desired_img)
            # print(time.perf_counter() - t)
            # bottleneck end
        else:
            # skip pose detection for this step
            achieved_pose = SimpleNamespace(pose_world_landmarks=[])
            desired_pose = SimpleNamespace(pose_world_landmarks=[])


        # get normalized goal distance
        achieved_obs = np.array([])

        if achieved_pose.pose_world_landmarks:
            # only first detected pose
            for landmark in achieved_pose.pose_world_landmarks[0]:
                achieved_obs = np.append(achieved_obs, (landmark.x, landmark.y, landmark.z))
            self.last_detected_obs_achieved = achieved_obs
        else:
            # no detected achieved pose. fallback...
            achieved_obs = self.last_detected_obs_achieved

        achieved_obs = self._normalize(achieved_obs, -1, 1)
        
        achieved_ob_height = superobs[0]
        achieved_ob_height = self._normalize(achieved_ob_height, 1.0, 2.0)
        achieved_obs = np.append(achieved_obs, achieved_ob_height)

        obs.extend(achieved_obs)
        

        desired_obs = np.array([])

        if desired_pose.pose_world_landmarks:
            for landmark in desired_pose.pose_world_landmarks[0]:
                desired_obs = np.append(desired_obs, (landmark.x, landmark.y, landmark.z))
            self.last_detected_obs_desired = desired_obs
        else:
            # no detected desired pose. fallback...
            desired_obs = self.last_detected_obs_desired

        desired_obs = self._normalize(desired_obs, -1, 1)

        desired_ob_height = 1.3
        desired_ob_height = self._normalize(desired_ob_height, 1.0, 2.0)
        desired_obs = np.append(desired_obs, desired_ob_height)

        goaldist = np.linalg.norm(achieved_obs - desired_obs, axis=-1)


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
                observation=np.array(obs),
                achieved_goal=np.array(goaldist),
                desired_goal=self.ep_reward_threshold,
            )

        if self.is_plot and self.ep_num_steps % cfg.General.STEPSKIP_PLOT == 0:
            achieved_img_annotated = draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
            desired_img_annotated = draw_landmarks_on_image(self.desired_img.numpy_view(), desired_pose)
            self.parallel_plot_queue.put((achieved_img_annotated, desired_img_annotated, achieved_pose, desired_pose))

        return dictobs


    # is also used by HER (multi-dim. args.)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:

        reward = (achieved_goal < desired_goal)

        if achieved_goal.ndim == 0:
            # single live step (no replay)

            if self.cfg.GoalRewardThreshold.IS_NUDGING:
                if achieved_goal < self.tr_goaldist_personal_best:
                    print(f'NEW PERSONAL BEST! {achieved_goal} > {self.tr_goaldist_personal_best}')
                    self.tr_goaldist_personal_best = achieved_goal
                    reward = np.bool_(True)

        return reward.astype(np.float64)
    

    def reset_model(self):
        obs_init = None

        if self.ep_num_steps > 0:
            # episode report
            print('ep_num_steps', self.ep_num_steps)
            print('ep_first_reward_step', self.ep_first_reward_step)
            print('ep_goal_desired', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved', self.ep_obs_cur['achieved_goal'])
            print('ep_goaldist_min', min(self.ep_goaldists))
            print('ep_goaldist_mean', np.mean(self.ep_goaldists))
            print('ep_goaldist_max', max(self.ep_goaldists))
            print('ep_reward_threshold', cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT, self.ep_reward_threshold)
            print('ep_traj_is_halved', self.ep_traj_is_halved)
            print('ep_rewards_mean', self.ep_rewards_mean)
            if self.ep_rewards_mean == 0:
                print('WARNING: zero-sum-ep. => wasted ep.')
                self.tr_total_zero_sum_eps += 1
                self.tr_total_zero_sum_eps_steps += self.ep_num_steps
            print('\n')

        if self.ep_num_steps > 1:
            if self.cfg.TrajectoryHalving.IS_ENABLED:
                ep_goaldist_min = min(self.ep_goaldists)
                if ep_goaldist_min < self.last_ep_goaldist_min:
                    obs_init = self._reset_half_episode(ep_goaldist_min)

        if not obs_init:
            obs_init = self._reset_full_episode()

        print('ep_init_goaldist', obs_init['achieved_goal'])
        return obs_init
    

    def _reset_full_episode(self):
        obs_init = super().reset_model()
        # self.desired_obs = self._get_desired_obs()
        self.last_ep_goaldist_min = np.inf
        self.last_ep_rewards_mean = 0
        self.ep_traj_is_halved = False
        # noisy relative threshold (varies by initial state noise)
        # self.ep_reward_threshold = self.cfg.GoalRewardThreshold.MAX_FRAC_DEFAULT * obs_init['achieved_goal']

        # 40k, 1, noPen:    worst, no learning /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f438218/le-pose-imitation-v4/16-36-44/rl_model_finished
        # 40k, 10, pen, no zero-eps.:   worse, no learning /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/20-23-38/rl_model_finished
        # 40k, 10, pen:     not better, collapses fast, trying /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/18-32-28/rl_model_finished
        # 40k, inf/none, noPen:     bad, but trying /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f438218/le-pose-imitation-v4/17-30-34/rl_model_finished
        # 40k, 10, noPen:   better, some standing and turning, some straight legs /home/t14/Documents/tuhh/dsf/Scilab-RL/data/f438218/le-pose-imitation-v4/16-10-26/rl_model_finished
        # 40k, none, pen:   better, some turning, one-legged /home/t500/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/17-37-41/rl_model_finished
        # 40k, none, pen, less zero-eps.:   good, reliable turning, one-legged, resemblence /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/19-57-30/rl_model_finished
        # 40k, none, pen, min. zero-eps.:   best, reliable turning, hand moves up /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/20-59-07/rl_model_finished
        # 100k,                         : falling to knees (more stable position? does not know ground/height (ob dim./sense))  /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/20-59-07_restored/rl_model_finished
        # 200k,                         : barely pos. rewards anymore, rather tries to sit down (only avoids falling / penalties), no resemblence anymore /home/t14/Documents/tuhh/dsf/Scilab-RL/data/835ce73/le-pose-imitation-v4/20-59-07_restored_restored/rl_model_finished
        CONSECUTIVE_NEG_FEPS_UNTIL_PERSONAL_RESET = 10
        if self.tr_goaldist_personal_best < 0: # or self.tr_feps_consecutive_neg >= CONSECUTIVE_NEG_FEPS_UNTIL_PERSONAL_RESET:
            print('reset personal best.')
            self.tr_goaldist_personal_best = np.inf

        if self.cfg.GoalRewardThreshold.IS_ADAPTIVE:
            self.ep_reward_threshold = (1 - self.ep_rewards_mean) * (obs_init['achieved_goal'])

        self.tr_feps_total += 1
        if self.fep_rewards_sum < 0:
            self.tr_feps_consecutive_neg += 1
        else:
            self.tr_feps_consecutive_neg = 0

        print('tr_feps_total', self.tr_feps_total)
        print('tr_feps_consecutive_neg', self.tr_feps_consecutive_neg)
        print('tr_total_zero_sum_eps', self.tr_total_zero_sum_eps)
        print('tr_total_zero_sum_eps_steps', self.tr_total_zero_sum_eps_steps)
        print('tr_goaldist_personal_best', self.tr_goaldist_personal_best)
        print('fep_rewards_sum', self.fep_rewards_sum)
        self.fep_rewards_sum = 0

        self._reset()
        return obs_init


    def _reset_half_episode(self, ep_goaldist_min):
        idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
        print('halving!', idx_halving)
        qpos, qvel = self.ep_states[idx_halving]
        # qpos, qvel = self._add_noise(qpos, qvel)
        self.set_state(qpos, qvel)
        self.ep_traj_is_halved = True
        self.last_ep_rewards_mean = self.ep_rewards_mean
        self.last_ep_goaldist_min = ep_goaldist_min

        obs_init = self._get_obs()
        self._reset()
        return obs_init


    def _reset(self):
        self.ep_rewards_mean: float = -1
        self.ep_num_steps: int = 0
        self.ep_first_reward_step: int = -1
        self.ep_obs_cur = None
        self.ep_goaldists = []
        self.ep_states = []
        self.ep_is_perfect = False
        # print('desired_obs', self.desired_obs)

        # landmarker reset: better/correct detection of start pose
        # TODO wait for efficient reset() implementation in API
        # workaround by re-create
        if self.landmarker_achieved:
            self.landmarker_achieved.close()
            del self.landmarker_achieved
            self.landmarker_achieved = None
        if self.landmarker_desired:
            self.landmarker_desired.close()
            del self.landmarker_desired
            self.landmarker_desired = None

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


    def _normalize(self, val, min_val, max_val):
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
        extplot.set_xlim3d(-1, 1)
        extplot.set_ylim3d(-1, 1)
        extplot.set_zlim3d(1, -1) # flip z-axis 

        for group in LANDMARK_GROUPS:
            if achieved_pose.pose_world_landmarks:
                plotX = [achieved_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [achieved_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [achieved_pose.pose_world_landmarks[0][i].z for i in group]
                extplot.plot(plotX, plotZ, plotY, color='red')

            if desired_pose.pose_world_landmarks:
                plotX = [desired_pose.pose_world_landmarks[0][i].x for i in group]
                plotY = [desired_pose.pose_world_landmarks[0][i].y for i in group]
                plotZ = [desired_pose.pose_world_landmarks[0][i].z for i in group]
                extplot.plot(plotX, plotZ, plotY, color='green')
        
        extplot.draw(extplot.get_figure().canvas.get_renderer())
        plt.pause(0.00001)
