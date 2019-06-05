import sys
sys.path.append("/usr/local/lib")
import time
import pyrealsense2 as rs
import numpy as np
# import cv2
from donkeycar.parts.camera import BaseCamera


class RealSenseCamera(BaseCamera):
    def __init__(self, resolution=(848, 800), framerate=20):
        cfg = rs.config()
        self.pipeline = rs.pipeline()

        cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
        cfg.enable_stream(rs.stream.fisheye, 2) # Right camera

        self.pipeline.start(cfg)
        self.frame = None
        self.on = True

        print('RealSense Camera loaded... warming up camera')
        time.sleep(2)

    def run(self):
        frames = self.pipeline.wait_for_frames()
        
        left = frames.get_fisheye_frame(1)
        left_frame = np.asanyarray(left.get_data())
        # frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2RGB)
        return left_frame


    def update(self):
        while True:
            self.frame = self.run()

            if not self.on:
                break


    def shutdown(self):
        self.on = False
        print('Stopping RealSense Camera')
        self.pipeline.stop()

