from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default="/home/cyc/pycharm/vGesture/checkpoints/yolo_human_best.trt", help="TRT engine Path")
    parser.add_argument("-i", "--image", default="/home/cyc/pycharm/vGesture/test_img/video_0/rgb/0000.png", help="image path")
    parser.add_argument("-o", "--output", default="/home/cyc/pycharm/vGesture/lib/core/test_img/1123/trt_testimg.png", help="image output path")
    parser.add_argument("-v", "--video", help="video path or camera index")
    parser.add_argument("--end2end", default=True, action="store_true", help="use end2end engine")


    args = parser.parse_args()
    print(args)
    
    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
      sum = []
      for i in range(0,100):
        t0=time.time()
        origin_img = pred.inference(img_path, conf=0.8, end2end=args.end2end)
        sum.append((time.time()-t0))
      print('per pic:',np.mean(sum))
      print('fps:',1/np.mean(sum))
      cv2.imwrite("%s" %args.output , origin_img)
    if video:
      pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam
