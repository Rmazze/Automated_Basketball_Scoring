import math
import os
import sys

import cv2
import numpy as np

from utility.stat_utility import *


def get_overlap(roi1, roi2):
    #roi1 is intended to be the basket bounding box
    def compute_area(roi):
        _, _, w, h = roi
        return w*h

    def get_insersec_region(roi1, roi2):
        roi1_min_x, roi1_min_y, roi1_w, roi1_h = roi1
        roi1_max_x = roi1_min_x + roi1_w
        roi1_max_y = roi1_min_y + roi1_h
        roi2_min_x, roi2_min_y, roi2_w, roi2_h = roi2
        roi2_max_x = roi2_min_x + roi2_w
        roi2_max_y = roi2_min_y + roi2_h

        intersect = True
        if roi1_min_x > roi2_max_x or roi1_max_x < roi2_min_x:
            intersect = False
        if roi1_min_y > roi2_max_y or roi1_max_y < roi2_min_y:
            intersect = False

        if not intersect:
            return None
        min_x = max(roi1_min_x, roi2_min_x)
        max_x = min(roi1_max_x, roi2_max_x)
        min_y = max(roi1_min_y, roi2_min_y)
        max_y = min(roi1_max_y, roi2_max_y)

        inter_w = max_x - min_x
        inter_h = max_y - min_y

        return [min_x, min_y, inter_w, inter_h]

    roi_inter = get_insersec_region(roi1, roi2)

    if roi_inter != None:
        area_1 = compute_area(roi1)
        area_inter = compute_area(roi_inter)

        overlap = area_inter/area_1
    else:
        overlap = 0

    return overlap

class Tracker:

    def __init__(self):
        self.initBB = None
        self.tracked_player = (0, 0, 0, 0)
        
        self.tracker = cv2.TrackerCSRT_create() 
        
        self.history_distance = []

        self.to_destroy = False
        
    def track(self, frame, boxes, scores, frame_id):
        (H, W) = frame.shape[:2]
        #draw bounding box of player to track
        if len(boxes)>0:
            coor = boxes[0] #return only the first bounding box -> since player nearer to the ball
            p1 = (int(coor[0]), int(coor[1]))
            p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 8, 5)

	    #caso 1: tracker not yet initialized, no initial bounding box... initialize tracker with the first detection available
        if self.initBB is None:            
            
            self.tracker = cv2.TrackerCSRT_create()
            
            print ("waiting for the first detection to inizialize the tracker!")
            for i, bbox in enumerate(boxes):
                if scores[i]>0.40: #initialize the tracker only if the confidence is high enough         
                    coor = np.array(bbox[:4], dtype=np.int32)
                    self.initBB = (coor[0], coor[1], coor[2], coor[3])
                    self.tracked_player = (int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]))

                    self.tracker.init(frame, self.initBB)
                
	    #there is detection -> update tracker 
        if self.initBB is not None: 
            (success, tracked_box) = self.tracker.update(frame)

            if success:
                p1 = (int(tracked_box[0]), int(tracked_box[1]))
                p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
                cv2.rectangle(frame, p1, p2, (255,255,0), 6, 3)
            
                self.tracked_player = (int(tracked_box[0]), int(tracked_box[1]), int(tracked_box[2]), int(tracked_box[3]))
               
            else: 
                self.initBB = None
                print("tracking ha perso il targhet! in attesa di reinizializzazione")
        
        
        #caso 2 detection and tracker are available... compute euclidean distance between detection and tracking
        if (self.initBB is not None and len(boxes)>0):
            p1_tracker = (int(tracked_box[0]), int(tracked_box[1]))
            p2_tracker = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
            center_tracker = (int((p1_tracker[0]+p2_tracker[0])/2),int((p1_tracker[1]+p2_tracker[1])/2))
            bbox = boxes[0]
            coor = np.array(bbox[:4], dtype=np.int32)
            p1_prediction = (int(coor[0]), int(coor[1]))
            p2_prediction = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
            center_prediction = (int((p1_prediction[0]+p2_prediction[0])/2),int((p1_prediction[1]+p2_prediction[1])/2))
            
            distance = math.sqrt((center_tracker[0]-center_prediction[0])**2 + (center_tracker[1]-center_prediction[1])**2)
            self.history_distance.append(distance*scores[0])
            
            # if tracking is more distant than a treshold for at least 2 frames or the ball more distant than a threshold for 3 frames reinitialize the tracker
            if len(self.history_distance) >= 2:
                if (self.history_distance[-1]>20 and self.history_distance[-2]>30):

                    self.history_distance[-1] = 0
                    self.history_distance[-2] = 0

                    print("tracker reinitialization")
                    self.tracker = cv2.TrackerCSRT_create()
                    
                    for i, bbox in enumerate(boxes):
                        coor = np.array(bbox[:4], dtype=np.int32)
                        self.initBB = (coor[0], coor[1], coor[2], coor[3])
                        p1 = (int(coor[0]), int(coor[1]))
                        p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
                        frame = cv2.rectangle(frame, p1, p2, (255,0,100), 6, 3)
                        self.tracker.init(frame, self.initBB)

                        self.tracked_player = (int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]))

                    #self.to_destroy=True


class Trackers:

    def __init__(self):
        self.tracker_list = []

    def track(self, frame, team_boxes, team_numbers, team_scores, f, frame_id):
        (H, W) = frame.shape[:2]
        image = frame.copy()

        survived_trackers = []
        
        for idx1 in range(len(self.tracker_list)):
            for idx2 in range(idx1+1, len(self.tracker_list)):
                tracker1 = self.tracker_list[idx1]
                tracker2 = self.tracker_list[idx2]
                #if tracker1["team_num"] == tracker2["team_num"]: #treshold to use 0.4
                overlap1 = get_overlap(tracker1["tracker"].tracked_player, tracker2["tracker"].tracked_player)
                overlap2 = get_overlap(tracker2["tracker"].tracked_player, tracker1["tracker"].tracked_player)
                if overlap1 >= 0.8 or overlap2 >= 0.8:
                    tracker1["tracker"].to_destroy = True

        for tracker in self.tracker_list:
            if not tracker["tracker"].to_destroy:
                survived_trackers.append(tracker)

        self.tracker_list = survived_trackers 

        for tracker in self.tracker_list:
            box = tracker["tracker"].tracked_player
            if tracker["team_num"] == 0:
                image = draw_rect(image, box, (255, 0, 0))
            elif tracker["team_num"] == 1:
                image = draw_rect(image, box, (0, 255, 0))
            else:
                image = draw_rect(image, box, (0, 0, 255))

            f.write('{},-1,{},{},{},{},{},-1,-1,-1,{}\n'.format(frame_id, int(box[0]), int(box[1]), int(box[2]), int(box[3]), 0.99, tracker["team_num"])) #0.99 is the score
            tracker["used"] = False

        for (team_box, team_num, team_score) in zip(team_boxes, team_numbers, team_scores):
            
            #variable that tell us if we found a tracker that already track this box
            best_match = None
            for x in self.tracker_list:
                if not x["used"]:
                    if x["team_num"] == team_num:
                        dist = distance_boxes(x["tracker"].tracked_player, team_box)
                        #if dist < 50:
                        #print("match found, dist: ", dist)
                        if best_match is None:
                            best_match = {"elem": x, "dist": dist}
                        else:
                            if dist < best_match["dist"]:
                                best_match = {"elem": x, "dist": dist}

            if best_match is not None:
                tracker = best_match["elem"]
                tracker["used"] = True
                tracker["tracker"].track(frame, [team_box], [team_score], frame_id)
            else:
                new_tracker = Tracker()
                new_tracker.track(frame, [team_box], [team_score], frame_id)
                self.tracker_list.append({"tracker": new_tracker, "team_num": team_num, "used": True})
        
        for tracker in self.tracker_list:
            if not tracker["used"]: 
                tracker["tracker"].track(frame, [], [], frame_id)
                tracker["used"]=True
        
        return image


def opencv_tracking(video_path, team_detection_path, out_txt_file):
    #gt_dict = get_dict(detection_path)
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
    
    trackers = Trackers()
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while ret:
        print("Frame: {}".format(frame_id))
        ret, frame = video.read()

        if not ret:
            continue

        #boxes, scores, names = [], [], []
        #boxes, scores, names, complete = get_gt(frame_id, team_dict)

        boxes_team, scores_team, names_team, team_numbers = (
            [[0, 0, 0, 0]],
            [[0]],
            [[0]],
            [[0]],
        )
        
        boxes_team, scores_team, names_team, complete_team, team_numbers = get_gt(frame, frame_id, team_dict)
        
        image = trackers.track(frame, boxes_team, team_numbers, scores_team, f, frame_id)
         
        cv2.imshow("image", image)
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
    args.out_tracker = "./det_tracking/player_tracking/" + args.video_name + ".txt"

    opencv_tracking(args.video, args.det, args.out_tracker)
