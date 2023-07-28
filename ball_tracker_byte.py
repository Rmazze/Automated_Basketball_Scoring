import os
from re import X
import sys

import cv2
import numpy as np

import torch

from utility.stat_utility import *
from yolox.tracker.byte_tracker import BYTETracker
#from yolox.tracker.my_byte_tracker import BYTETracker

#va male, byte tracker non va bene per il tracking della palla, servono più detections

def get_gt(frame_id, gt_dict):
    if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
        return [], [], [], []
        
    frame_info = gt_dict[frame_id]
    
    detections = []
    out_scores = []
    ids = []
    complete = []
    
    for i in range(len(frame_info)):
        coords = frame_info[i]['coords']
        
        x1,y1,w,h = coords

        detections.append([x1,y1,w,h])
        out_scores.append(frame_info[i]['conf'])
        ids.append(1)

        complete.append([x1,y1,w,h,frame_info[i]['conf'],1])

    return detections, out_scores, ids, complete


# dato il file MOTA ottieni il dizionario contenente tutte le detection:
def get_dict(filename):
    with open(filename) as f:    
        d = f.readlines()
        
    d = list(map(lambda x:x.strip(),d))
    
    last_frame = int(d[-1].split(',')[0])
    gt_dict = {x:[] for x in range(last_frame+1)}
    
    for i in range(len(d)):
        a = list(d[i].split(','))
        a = list(map(float,a))    
        
        coords = a[2:6]
        confidence = a[6]

        gt_dict[a[0]].append({'coords':coords,'conf':confidence})
        
    return gt_dict

#def make_parser():
#    parser = argparse.ArgumentParser("YOLOX Eval")
#    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
#    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
#
#    # distributed
#    parser.add_argument(
#        "--dist-backend", default="nccl", type=str, help="distributed backend"
#    )
#    parser.add_argument(
#        "--dist-url",
#        default=None,
#        type=str,
#        help="url used to set up distributed training",
#    )
#    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
#    parser.add_argument(
#        "-d", "--devices", default=None, type=int, help="device for training"
#    )
#    parser.add_argument(
#        "--local_rank", default=0, type=int, help="local rank for dist training"
#    )
#    parser.add_argument(
#        "--num_machines", default=1, type=int, help="num of node for training"
#    )
#    parser.add_argument(
#        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
#    )
#    parser.add_argument(
#        "-f",
#        "--exp_file",
#        default=None,
#        type=str,
#        help="pls input your expriment description file",
#    )
#    parser.add_argument(
#        "--fp16",
#        dest="fp16",
#        default=False,
#        action="store_true",
#        help="Adopting mix precision evaluating.",
#    )
#    parser.add_argument(
#        "--fuse",
#        dest="fuse",
#        default=False,
#        action="store_true",
#        help="Fuse conv and bn for testing.",
#    )
#    parser.add_argument(
#        "--trt",
#        dest="trt",
#        default=False,
#        action="store_true",
#        help="Using TensorRT model for testing.",
#    )
#    parser.add_argument(
#        "--test",
#        dest="test",
#        default=False,
#        action="store_true",
#        help="Evaluating on test-dev set.",
#    )
#    parser.add_argument(
#        "--speed",
#        dest="speed",
#        default=False,
#        action="store_true",
#        help="speed test only.",
#    )
#    parser.add_argument(
#        "opts",
#        help="Modify config options using the command-line",
#        default=None,
#        nargs=argparse.REMAINDER,
#    )
#    # det args
#    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
#    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
#    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#    parser.add_argument("--tsize", default=None, type=int, help="test img size")
#    parser.add_argument("--seed", default=None, type=int, help="eval seed")
#    # tracking args
#    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
#    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
#    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#    return parser

class FakeArgs():
    def __init__(self) -> None:
        self.mot20 = False
        self.track_thresh = 0.2
        self.track_buffer = 30
        self.match_thresh = 0.1

class Tracker:

    def __init__(self):
        self.detections = []
        self.tracked_ball = None
        
        self.tracker = BYTETracker(FakeArgs())

    def convert_bbox_xywh_to_x1y1x2y2(self, bbox):
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        return x, y, x2, y2

    def convert_bbox_x1y1x2y2_to_xywh(self, bbox_id):
        x1, y1, x2, y2, id = bbox_id
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h], id 

    def track(self, frame, boxes, scores, f, frame_id):
        self.detections = []
        (H, W) = frame.shape[:2]
        if len(boxes) > 0:
            x, y, x2, y2 = self.convert_bbox_xywh_to_x1y1x2y2(boxes[0])
            self.detections = np.array([[x,y,x2,y2,scores[0]]])
        else:
            #self.detections = np.array([[1. ,1. ,1. ,1. ,0.]])
            self.detections = np.zeros(shape=(5,5))
        self.tracked_ball = self.tracker.update(torch.from_numpy(self.detections), img_info = (H,W), img_size = (H,W))

        print("self.tracked_ball: ", self.tracked_ball)

        if len(self.tracked_ball) > 0:
            ball_box = self.tracked_ball[0]
            frame = draw_rect(frame, ball_box, (0, 255, 255))  
            f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame_id, int(ball_box[0]), int(ball_box[1]), int(ball_box[2]), int(ball_box[3]) , 1))            
        return frame



def opencv_tracking(video_path, detection_path, out_txt_file):
    gt_dict = get_dict(detection_path)
    frame_id = 1

    try:
        os.remove(out_txt_file)
    except :
        None
    f = open(out_txt_file, "a")
    

    # Input
    video = cv2.VideoCapture(video_path)

    # Output
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #doutput video format mp4
    out = cv2.VideoWriter('output-finale.mp4', fourcc, 30.0, (int(video.get(3)), int(video.get(4))), True) #output video properties definition
   
    
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    ret = True
    
    #trackers = Trackers()
    tracker = Tracker()
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while ret:
        print("Frame: {}".format(frame_id))
        ret, frame = video.read()

        if not ret:
            continue

        boxes, scores, names = [], [], []
        boxes, scores, names, complete = get_gt(frame_id,gt_dict)
        
        frame = tracker.track(frame, boxes, scores, f, frame_id)

         
        cv2.imshow("image", frame)
        cv2.waitKey(1)
        out.write(frame)    

        frame_id+=1

    out.release()

############################################################
#  Main Tracking
############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--video_name', required=True,
                        metavar="video name",
                        help='Video to apply the tracking on')
    args = parser.parse_args()

    args.video = "./input_video/" + args.video_name + ".mp4"
    args.det = "./det_tracking/ball_detection_yolo/" + args.video_name + ".txt"
    args.out_tracker = "./det_tracking/ball_tracking_byte/" + args.video_name + ".txt"

    opencv_tracking(args.video, args.det, args.out_tracker)
