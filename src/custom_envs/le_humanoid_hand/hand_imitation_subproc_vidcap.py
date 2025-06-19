import numpy as np
import multiprocessing
import glob
import cv2
from matplotlib import image

import git
PATH_GIT_WORKING_DIR = git.Repo('.', search_parent_directories=True).working_tree_dir

import logging
LOG = logging.getLogger(__name__)
LOG.propagate = False
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter(fmt='%(message)s'))
LOG.addHandler(consoleHandler)
multiprocessing.log_to_stderr(logging.DEBUG)


class VidCapSingletonSubprocess:

    parallel_vidcap_queue = multiprocessing.Queue()


    @staticmethod
    def parallel_vidcap(queue: multiprocessing.Queue):
        vidcap = cv2.VideoCapture(0)
        while True:
            desired_imgpaths = glob.glob(PATH_GIT_WORKING_DIR + '/mediapipe/poses/hand/*.jpg')
            if desired_imgpaths:
                desired_imgpath = desired_imgpaths[np.random.randint(len(desired_imgpaths))]
                try:
                    frame = image.imread(desired_imgpath)
                except FileNotFoundError:
                    LOG.warning('gesture image not found.')
            else:
                success, frame = vidcap.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    LOG.warning('video camera unavailable.')
                    break

            if queue.empty():
                queue.put_nowait((frame))
        vidcap.release()



multiprocessing.Process(target=VidCapSingletonSubprocess.parallel_vidcap, args=((VidCapSingletonSubprocess.parallel_vidcap_queue,)), daemon=True).start()
