import multiprocessing.synchronize
import numpy as np
import multiprocessing
import glob
import cv2
import time
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


class VidCapSingletonProc:

    parallel_vidcap_queue = multiprocessing.Queue(maxsize=1)
    change_img_ev = multiprocessing.Event()
    change_img_ev.set()
    autochange_ev = multiprocessing.Event()
    autochange_interval_sec = 10

    @staticmethod
    def parallel_vidcap(queue: multiprocessing.Queue, change_img_ev: multiprocessing.synchronize.Event):
        vidcap = cv2.VideoCapture(0)
        imgpath_current = None
        last_change_timesec = time.time()

        while True:
            desired_imgpaths = glob.glob(PATH_GIT_WORKING_DIR + '/mediapipe/poses/hand/*.jpg')
            if desired_imgpaths:
                is_autochange_triggered = (VidCapSingletonProc.autochange_ev.is_set() and (time.time() - last_change_timesec) > VidCapSingletonProc.autochange_interval_sec)
                if change_img_ev.is_set() or is_autochange_triggered:
                    desired_imgpath = imgpath_current = desired_imgpaths[np.random.randint(len(desired_imgpaths))]
                    change_img_ev.clear()
                    last_change_timesec = time.time()
                else:
                    desired_imgpath = imgpath_current

                try:
                    frame = image.imread(desired_imgpath)                    
                except FileNotFoundError:
                    LOG.warning('gesture image not found.')
                    imgpath_current = desired_imgpaths[np.random.randint(len(desired_imgpaths))]

            else:
                success, frame = vidcap.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    LOG.warning('video camera unavailable.')
                    break

            if queue.empty():
                queue.put((frame))

        vidcap.release()


multiprocessing.Process(target=VidCapSingletonProc.parallel_vidcap,
                        args=((VidCapSingletonProc.parallel_vidcap_queue, VidCapSingletonProc.change_img_ev)), daemon=True).start()
