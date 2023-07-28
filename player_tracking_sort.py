import math
import os
from re import X
import sys

import cv2
import numpy as np

from utility.stat_utility import *
#from sort import *
from my_sort import *

class Tracker:

    def __init__(self):
        self.detections = []
        self.tracked_players_x1y1x2y2_id = None
        
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.6) #standard values: max_age=1, min_hits=3, iou_threshold=0.3
        
    def convert_bbox_xywh_to_x1y1x2y2(self, bbox):
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        return x, y, x2, y2

    def convert_bbox_x1y1x2y2_to_xywh(self, bbox_id_team):
        x1, y1, x2, y2, id, team = bbox_id_team
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h], id, team 

    def track(self, frame, boxes, team_numbers, scores, f, frame_id):
        self.detections = []
        for bbox, score, team in zip (boxes, scores, team_numbers):
            x1, y1, x2, y2 = self.convert_bbox_xywh_to_x1y1x2y2(bbox)
            self.detections.append([x1, y1, x2, y2, score, team])
        if len(self.detections) == 0:
            self.detections = np.empty((0, 6))
        else:
            self.detections = np.array(self.detections)

        self.tracked_players_x1y1x2y2_id = self.tracker.update(self.detections)
        print("self.tracked_players_x1y1x2y2_id: ", self.tracked_players_x1y1x2y2_id)

        for bbox_id in self.tracked_players_x1y1x2y2_id:
            bbox, id, team = self.convert_bbox_x1y1x2y2_to_xywh(bbox_id)
            f.write('{},-1,{},{},{},{},{},-1,-1,-1,{}\n'.format(frame_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), score, team))

            if team == 1:
                color = (0, 255, 0)
            elif team == 2:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0) 
            frame = draw_rect(frame, bbox, color)     
        
        return frame



def opencv_tracking(video_path, team_detection_path, out_txt_file):
    team_dict = get_dict(team_detection_path)
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
    
    tracker = Tracker()
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while ret:
        print("Frame: {}".format(frame_id))
        ret, frame = video.read()

        if not ret:
            continue

        boxes_team, scores_team, names_team, team_numbers = (
            [[0, 0, 0, 0]],
            [[0]],
            [[0]],
            [[0]],
        )
        
        boxes_team, scores_team, names_team, complete_team, team_numbers = get_gt(frame, frame_id, team_dict)
        
        #image = trackers.track(frame, boxes_team, team_numbers, scores_team, f, frame_id)
        frame = tracker.track(frame, boxes_team, team_numbers, scores_team, f, frame_id)

         
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
    args.det = "./det_tracking/player_detection/" + args.video_name + ".txt"
    args.out_tracker = "./det_tracking/player_tracking_sort/" + args.video_name + ".txt"

    opencv_tracking(args.video, args.det, args.out_tracker)
