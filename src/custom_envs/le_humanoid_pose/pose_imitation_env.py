
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import BaseMujocoEnv
from ..le_base import base_practice_cfg

from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib import image

from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


TOTAL_OBSERVATION_FEATURES = 99

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

# TODO do we need 3d body-relative (world) landmarks? (instead of 2d canvas-relative image coords (normalized))
# https://github.com/google-ai-edge/mediapipe/issues/5325
# https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
# TODO reduce goal features?
# TODO 3d plot of landmarks in pyplot (instead of overlay)?
# TODO fix pose landmarker memory leak


class PoseImitationEnv(HumanoidEnv):

    def __init__(self):
        # TODO extract hyperparams
        HumanoidEnv.__init__(self, exclude_current_positions_from_observation=True, width=480, height=480)
        # self.frame_skip = 10
        
        self.cfg: base_practice_cfg = base_practice_cfg
        self.desired_img = image.imread('/home/t14/Documents/tuhh/dsf/Scilab-RL/mediapipe/poses/pose1.jpg')
        self.desired_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.desired_img.copy())

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
        self.landmarker_options_achieved = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='/home/t14/Documents/tuhh/dsf/Scilab-RL/mediapipe/model/pose_landmarker_lite.task',            
                # cpu vs gpu
                # https://forums.developer.nvidia.com/t/how-to-install-opengl-libs-of-nvidia/175409
                # https://stackoverflow.com/questions/77707532/how-to-check-for-and-enforce-gpu-usage-for-mediapipe-frame-processing/79202595#79202595
                # prime-select nvidia
                # glxinfo | grep -i opengl
                # MUJOCO_GL=egl %python ...% (faster than Delegate.GPU)
                delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1)

        self.landmarker_options_desired = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='/home/t14/Documents/tuhh/dsf/Scilab-RL/mediapipe/model/pose_landmarker_lite.task',            
                delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1)
        

        if self.cfg.General.IS_OBSERVATION_GOAL_EXTENDED:
            obspace_shape = (TOTAL_OBSERVATION_FEATURES * 2,)
        else:
            obspace_shape = (TOTAL_OBSERVATION_FEATURES,)

        observation_space = spaces.Box(-np.inf, np.inf, shape=obspace_shape, dtype='float64')
        practice_space = spaces.Box(-np.inf, np.inf, shape=(TOTAL_OBSERVATION_FEATURES,), dtype='float64')

        # https://scilab-rl.github.io/Scilab-RL/wiki/Add-environment-to-MakeDictObs-wrapper.html
        self.observation_space = spaces.Dict(
            dict(
                observation=observation_space,
                desired_goal=practice_space,
                achieved_goal=practice_space,
            )
        )

        # once
        self.last_ep_rewards_mean: float = 0
        self.last_ep_goal_distance_min_normed: float = np.inf
        self.ep_num_steps: int = 0
        # self.desired_goal = self.cfg.PracticeSpace.d[2]
        self.last_detected_pose_achieved = np.full(TOTAL_OBSERVATION_FEATURES, 1)
        self.last_detected_pose_desired = np.full(TOTAL_OBSERVATION_FEATURES, 1)

        self.landmarker_achieved = None
        self.landmarker_desired = None

        self.extplot = None

        self._reset_episode()
        print('le-walker-2d initialized.')


    def compute_reward(
        self, achieved_goal_normed: np.ndarray, desired_goal_normed: np.ndarray, info
    ) -> float:

        # goaldiff_weighted = self.cfg.PracticeSpace.d[3] * np.array([achieved_goal_normed - desired_goal_normed])
        goaldiff_weighted = np.array([achieved_goal_normed - desired_goal_normed])

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

        self.ep_rewards_mean = ((self.ep_num_steps * self.ep_rewards_mean) + reward) / (self.ep_num_steps + 1)
        self.ep_num_steps += 1

        terminated = False
        truncated = False

        # termination shaping?
        # faster learning: decrease search/interaction space (find terminations (=constraints))
        # imitation vs. direction (guidance, experience, coaching)
        # TODO how to recognize/mitigate destructive terminations? (lead to impossible goals/searches)
        # TODO should all constraints also be practiced? (ie. as dim. in practice (multi-)goalspace, not only in general obs., "conscious about constraints")
        dims_outside, = np.where(np.logical_or(obs['achieved_goal'] < 0, obs['achieved_goal'] > 1))
        
        # if len(dims_outside) > 0:
        #     print('outside practice space!', obs['achieved_goal'][dims_outside])
        #     terminated = self.cfg.PracticeSpace.IS_TERMINATION_IF_OUTSIDE
        #     reward = self.cfg.PracticeSpace.REWARD_IF_OUTSIDE

        nose_y = obs['achieved_goal'][1] # inverted height (nose)
        if nose_y > -0.4:
            print('fell down.', nose_y)
            terminated = True

        if self.ep_num_steps > self.cfg.General.EPISODE_TRUNCATION_STEPS_MAX:
            print('truncated.')
            info['success'] = bool(self.ep_rewards_mean > self.cfg.General.EPISODE_SUCCESS_THRESHOLD_REWARD_MEAN)
            truncated = True

        result = obs, float(reward[0]), terminated, truncated, info
        return result


    def reset_model(self):
        obs_init = None

        if self.ep_obs_cur:
            # print(self.ep_goal_distances)
            print('ep_num_steps', self.ep_num_steps)
            print('ep_first_reward_step', self.ep_first_reward_step)
            # print('ep_goal_distance_min_normed', min(self.ep_goal_distances_normed) / self.cfg.PracticeSpace.radius_normed)
            print('ep_goal_distance_min', min(self.ep_goal_distances_normed))
            # print('ep_goal_convergence_mean_per_step_normed', ((max(self.ep_goal_distances_normed) - min(self.ep_goal_distances_normed)) / self.ep_num_steps) / self.cfg.PracticeSpace.radius_normed)
            print('ep_goal_desired_normed', self.ep_obs_cur['desired_goal'])
            print('ep_goal_achieved_normed_end', self.ep_obs_cur['achieved_goal'])
            # print('ep_goal_reward_threshold_normed', self.ep_goal_reward_threshold_normed / self.cfg.PracticeSpace.radius_normed)
            print('ep_traj_is_halved', self.ep_traj_is_halved)
            print('ep_rewards_mean', self.ep_rewards_mean)
            print('\n')

        if self.ep_num_steps > 1:
            ep_goal_distance_min_normed = min(self.ep_goal_distances_normed)

            if self.cfg.TrajectoryHalving.IS_ENABLED and (ep_goal_distance_min_normed < self.last_ep_goal_distance_min_normed):
                idx_halving = self._get_idx_for_trajectory_halving(self.cfg.TrajectoryHalving.STRAT)
                print('halving!', idx_halving)
                qpos, qvel = self.ep_states[idx_halving]
                qpos, qvel = self._add_noise(qpos, qvel)

                self.set_state(qpos, qvel)
                self.ep_traj_is_halved = True
                self.last_ep_rewards_mean = self.ep_rewards_mean
                self.last_ep_goal_distance_min_normed = ep_goal_distance_min_normed
                obs_init = self._get_obs()

        if not obs_init:
            obs_init = super().reset_model()
            # self.desired_goal = self._new_goal()
            self.last_ep_goal_distance_min_normed = np.inf
            self.last_ep_rewards_mean = 0
            self.ep_traj_is_halved = False

        self._reset_episode()
        return obs_init


    def _get_obs(self):
        # render/detect pose only every nth frame
        # TODO may already with self.frame_skip param?
        if self.ep_num_steps % 1 == 0:
            # renders only rgb (cant render multiple modes simultanously)
            self.render_mode = 'rgb_array'
            
            # bottleneck start
            achieved_img = self.render().copy()
            achieved_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=achieved_img)

            # https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker#detect_for_video
            achieved_pose = self.landmarker_achieved.detect_for_video(achieved_img, self.ep_num_steps)
            desired_pose = self.landmarker_desired.detect(self.desired_img)
            # bottleneck start

        else:
            # TODO take last valid pose? extrapolate from last valid poses?
            achieved_pose = SimpleNamespace(pose_world_landmarks=[])
            desired_pose = SimpleNamespace(pose_world_landmarks=[])

        # achieved_goal_norm = self._normalize(self.get_achieved_goal(superobs), self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])
        # desired_goal_norm = self._normalize(self.desired_goal, self.cfg.PracticeSpace.d[0], self.cfg.PracticeSpace.d[1])

        achieved_goal_norm = []
        if achieved_pose.pose_world_landmarks:
            # multiple poses found? only first
            for landmark in achieved_pose.pose_world_landmarks[0]:
                achieved_goal_norm.append(landmark.x)
                achieved_goal_norm.append(landmark.y)
                achieved_goal_norm.append(landmark.z)
            self.last_detected_pose_achieved = achieved_goal_norm
            
        desired_goal_norm = []
        if desired_pose.pose_world_landmarks:
            # multiple poses found? only first
            for landmark in desired_pose.pose_world_landmarks[0]:
                desired_goal_norm.append(landmark.x)
                desired_goal_norm.append(landmark.y)
                desired_goal_norm.append(landmark.z)
            self.last_detected_pose_desired = desired_goal_norm

        # both must be detected, else fallback
        if not achieved_pose.pose_world_landmarks or not desired_pose.pose_world_landmarks:
            achieved_goal_norm = self.last_detected_pose_achieved
            desired_goal_norm = self.last_detected_pose_desired



        if False and self.ep_num_steps % 10 == 0:
            # TODO plot in separate thread?

            if not self.extplot:
                # once
                fig = plt.figure()
                ax2 = fig.add_subplot(131)
                self.plot_desired = ax2.imshow(np.zeros((1,1,3)))
                ax1 = fig.add_subplot(132)
                self.plot_achieved = ax1.imshow(np.zeros((1,1,3)))
                self.extplot = fig.add_subplot(133, projection="3d")

            desired_img_annotated = self._draw_landmarks_on_image(self.desired_img.numpy_view(), desired_pose)
            self.plot_desired.set_data(desired_img_annotated)
            self.plot_desired.draw(self.plot_desired.get_figure().canvas.get_renderer())

            achieved_img_annotated = self._draw_landmarks_on_image(achieved_img.numpy_view(), achieved_pose)
            self.plot_achieved.set_data(achieved_img_annotated)
            self.plot_achieved.draw(self.plot_achieved.get_figure().canvas.get_renderer())

            # plot topology connections
            # https://github.com/stebusse/mediapipe-plot-pose-live/blob/main/plot_pose_live.py
            self.extplot.clear()
            self.extplot.set_xlim3d(-1, 1)
            self.extplot.set_ylim3d(-1, 1)
            self.extplot.set_zlim3d(1, -1) # flip z-axis 

            for group in LANDMARK_GROUPS:
                if achieved_pose.pose_world_landmarks:
                    plotX = [achieved_pose.pose_world_landmarks[0][i].x for i in group]
                    plotY = [achieved_pose.pose_world_landmarks[0][i].y for i in group]
                    plotZ = [achieved_pose.pose_world_landmarks[0][i].z for i in group]
                    self.extplot.plot(plotX, plotZ, plotY, color='red')

                if desired_pose.pose_world_landmarks:
                    plotX = [desired_pose.pose_world_landmarks[0][i].x for i in group]
                    plotY = [desired_pose.pose_world_landmarks[0][i].y for i in group]
                    plotZ = [desired_pose.pose_world_landmarks[0][i].z for i in group]
                    self.extplot.plot(plotX, plotZ, plotY, color='green')
        
            self.extplot.draw(self.extplot.get_figure().canvas.get_renderer())

            plt.pause(0.00001)



        observation = []
        if self.cfg.General.IS_OBSERVATION_GOAL_EXTENDED:
            # TODO hash from desired_pose landmarks? (may ignore similarity/locality)
            observation = np.concatenate((achieved_goal_norm, desired_goal_norm))
        else:
            observation = achieved_goal_norm

        dictobs = dict(
                observation=np.array(observation),
                achieved_goal=np.array(achieved_goal_norm),
                desired_goal=np.array(desired_goal_norm),
            )

        return dictobs


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
        # TODO not normed yet
        self.ep_goal_distances_normed = []
        self.ep_states = []
        # TODO set norm
        self.ep_goal_reward_threshold_normed = 1.5
        self.ep_is_perfect = False
        # print('desired_goal', self.desired_goal)
        # print('goal tolerance', self.cfg.GoalRewardThreshold.MAX_FAC_DEFAULT * self.cfg.PracticeSpace.radius)

        if self.landmarker_achieved:
            self.landmarker_achieved.close()
        if self.landmarker_desired:
            self.landmarker_desired.close()

        self.landmarker_achieved = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_achieved)
        self.landmarker_desired = mp.tasks.vision.PoseLandmarker.create_from_options(self.landmarker_options_desired)


    def _get_idx_for_trajectory_halving(self, strat):
        idx_step = -1
        match strat:
            case self.cfg.TrajectoryHalving.Strat.HALF:
                idx_step = len(self.ep_states) // 2
            case self.cfg.TrajectoryHalving.Strat.HIGHEST_GOAL_CONVERGENCE:
                idx_step = np.argmin(np.gradient(self.ep_goal_distances_normed))
            case self.cfg.TrajectoryHalving.Strat.LOWEST_GOAL_DISTANCE:
                idx_step = np.argmin(self.ep_goal_distances_normed)
        return idx_step


    def _normalize(self, val, min_val, max_val):
        # manual normalization (obs fairness)
        # "interval-shifting"
        # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
        return (val - min_val) / (max_val - min_val)


    def _draw_landmarks_on_image(self, rgb_image, detection_result):
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
