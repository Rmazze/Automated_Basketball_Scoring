import math
import os
from statistics import mean
import sys

import cv2
import numpy as np

from utility.stat_utility import *


class Tracker:

    def __init__(self):
        self.initBB = None
        self.player_with_ball_tracking = (0, 0, 0, 0)
        
        self.tracker = cv2.TrackerCSRT_create() 
        
        self.history_distance = []
        self.history_distance_ball = []

        self.player_who_shot = None
        
        
    def track(self, frame, boxes, scores, frame_id, ball_box):
        (H, W) = frame.shape[:2]
        #draw bounding box of player with ball
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
                    self.player_with_ball_tracking = (int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]))

                    self.tracker.init(frame, self.initBB)
                
	    #there is detection -> update tracker 
        if self.initBB is not None: 
            (success, tracked_box) = self.tracker.update(frame)

            if success:
                p1 = (int(tracked_box[0]), int(tracked_box[1]))
                p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
                cv2.rectangle(frame, p1, p2, (255,255,0), 6, 3)
                
                #f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame_id, int(tracked_box[0]), int(tracked_box[1]), int(tracked_box[2]), int(tracked_box[3]) , 1))

                self.player_with_ball_tracking = (int(tracked_box[0]), int(tracked_box[1]), int(tracked_box[2]), int(tracked_box[3]))
               
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
            #print("scores[0]: ", scores[0])
            #print("d=" + str(distance)+ "    d*conficence: "+str(distance*scores[0]))
            self.history_distance.append(distance*scores[0])
            
            self.history_distance_ball.append(distance_boxes(tracked_box, ball_box))
            
            # if tracking is more distant than a treshold for at least 2 frames or the ball more distant than a threshold for 3 frames reinitialize the tracker
            if len(self.history_distance) >= 2 and len(self.history_distance_ball) >= 3:
                if ((self.history_distance[-1]>20 and self.history_distance[-2]>30) or 
                        (self.history_distance_ball[-1]>150 and self.history_distance_ball[-2]>150 and self.history_distance_ball[-3]>150)):

                    print("self.history_distance[-1]: ", self.history_distance[-1])
                    print("self.history_distance[-2]: ", self.history_distance[-2])  
                    self.history_distance[-1] = 0
                    self.history_distance[-2] = 0

                    print("self.history_distance_ball[-1]: ", self.history_distance_ball[-1])
                    print("self.history_distance_ball[-2]: ", self.history_distance_ball[-2])
                    print("self.history_distance_ball[-3]: ", self.history_distance_ball[-3])
                    self.history_distance_ball[-1] = 0
                    self.history_distance_ball[-2] = 0
                    self.history_distance_ball[-3] = 0

                    print("tracker reinitialization")
                    self.tracker = cv2.TrackerCSRT_create()
                    
                    for i, bbox in enumerate(boxes):
                        coor = np.array(bbox[:4], dtype=np.int32)
                        self.initBB = (coor[0], coor[1], coor[2], coor[3])
                        #f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame_id, int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]) , 1))
                        p1 = (int(coor[0]), int(coor[1]))
                        p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
                        frame = cv2.rectangle(frame, p1, p2, (255,0,100), 6, 3)
                        self.tracker.init(frame, self.initBB)

                        self.player_with_ball_tracking = (int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]))


class Statistics:
    def __init__(self):
        self.line_points = []
        self.resize = 1

        # variables for statistics 1:
        self.possesso_palla = np.array([0, 0, 0])
        self.ball_cumulative_position = np.array([0, 0])
        self.last_valid_ball = []
        self.storia_possesso_palla = []
        self.filtered_team_number = []  # team number with ball

        # per statistica 4
        self.history_mean_dist_team1 = []
        self.history_mean_dist_team2 = []

        self.ballDX = False
        self.ballSX = False
        self.history_distance_ball_center = []

        # per statistica 5:
        self.pressione = np.array([0, 0])

        # per statistica 6:
        self.right_ROI_up = (3010, 915, 50, 45)
        self.right_ROI_down = (3015, 965, 45, 40)
        self.left_ROI_up = (800, 990, 50, 45)
        self.left_ROI_down = (810, 1030, 45, 40)

        self.bg_right_up = None
        self.bg_right_down = None
        self.bg_left_up = None
        self.bg_left_down = None

        self.nzp_down_min = 50
        self.nzp_down_max = 300
        self.nzp_up_min = 130 #150

        self.delay_dx_up_val = 0
        self.delay_dx_down_min_val = 0
        self.delay_dx_down_max_val = 0
        self.delay_dx_up = 10
        self.delay_dx_down_min = 100
        self.delay_dx_down_max = 15

        self.delay_sx_up_val = 0
        self.delay_sx_down_min_val = 0
        self.delay_sx_down_max_val = 0
        self.delay_sx_up = 15 #10
        self.delay_sx_down_min = 100
        self.delay_sx_down_max = 15

        self.threshold = 45
        self.maxval = 255
        self.alpha = 0.1

        self.canestri_dx = 0
        self.canestri_sx = 0
        self.punti_dx = 0
        self.punti_sx = 0

        self.player_that_shot = None

        self.team1_counter_history = []
        self.team2_counter_history = []
        self.lunetta_history = []
        self.area_3pt_history = []

        self.player_with_ball_box = None #save the last position of the player with the ball
        self.path_mask3pt = "./input_video/3pt_mask.jpg"
        self.mask_3pt = cv2.imread(self.path_mask3pt, cv2.IMREAD_GRAYSCALE)
        #self.path_lunette_mask = "./input_video/lunette_piccole_mask.jpg"
        self.path_lunette_mask = "./input_video/lunette_grandi_mask.jpg"
        self.mask_lunette = cv2.imread(self.path_lunette_mask, cv2.IMREAD_GRAYSCALE)

    # initialization funct. to acquire half way line
    def initialize(self, img, resize):
        self.resize = resize

        cv2.namedWindow("selectpoint1")
        cv2.setMouseCallback("selectpoint1", self.draw_line)  

        (H, W) = img.shape[:2]

        global i
        i = cv2.resize(img, (int(W / 3), int(H / 3)))
        cv2.putText(
            i,
            "Select 2 extreme points of the middle line and press Q",
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        while True:
            # both windows are displaying the same img
            cv2.imshow("selectpoint1", i)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


        #initialize backgroud
        rux, ruy, ruw, ruh = self.right_ROI_up
        self.bg_right_up = img[ruy : ruy + ruh, rux : rux + ruw]
        rdx, rdy, rdw, rdh = self.right_ROI_down
        self.bg_right_down = img[rdy : rdy + rdh, rdx : rdx + rdw]

        #cv2.imshow("self.bg_right_up", self.bg_right_up)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        #cv2.imshow("self.bg_right_down", self.bg_right_down)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()

        lux, luy, luw, luh = self.left_ROI_up
        self.bg_left_up = img[luy : luy + luh, lux : lux + luw]
        ldx, ldy, ldw, ldh = self.left_ROI_down
        self.bg_left_down = img[ldy : ldy + ldh, ldx : ldx + ldw]

        #cv2.imshow("self.bg_left_up", self.bg_left_up)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        #cv2.imshow("self.bg_left_down", self.bg_left_down)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(i, (x, y), 2, (255, 0, 0), 2)
            self.line_points.append(
                (int((x * 3) / self.resize), int((y * 3) / self.resize))
            )

    def generate_file(self, f, frame_id):
        f.write("-----------------possesso palla:------------------- \n \n")
        txt1 = (
            "arbitro (frame/total_frame): "
            + str(self.possesso_palla[0])
            + " / "
            + str(frame_id)
            + "      ->   "
            + str(int(self.possesso_palla[0] / frame_id * 100))
            + "% \n"
        )
        f.write(txt1)
        txt2 = (
            "team A (frame/total_frame): "
            + str(self.possesso_palla[1])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.possesso_palla[1] / frame_id * 100))
            + "% \n"
        )
        f.write(txt2)
        txt3 = (
            "team B (frame/total_frame): "
            + str(self.possesso_palla[2])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.possesso_palla[2] / frame_id * 100))
            + "% \n"
        )
        f.write(txt3)

        f.write(
            "\n \n-----------------ball cumulative position:------------------- \n \n"
        )
        txt4 = (
            "SX (frame/total_frame): "
            + str(self.ball_cumulative_position[0])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.ball_cumulative_position[0] / frame_id * 100))
            + "% \n"
        )
        f.write(txt4)
        txt5 = (
            "DX (frame/total_frame): "
            + str(self.ball_cumulative_position[1])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.ball_cumulative_position[1] / frame_id * 100))
            + "% \n"
        )
        f.write(txt5)

        f.write("\n \n-----------------pressione difesa:------------------- \n \n")
        txt6 = (
            "affollamenti a SX: "
            + str(self.pressione[0])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.pressione[1] / np.sum(self.pressione) * 100))
            + "% \n"
        )
        f.write(txt6)
        txt7 = (
            "affollamenti a SX:: "
            + str(self.pressione[1])
            + " / "
            + str(frame_id)
            + "       ->   "
            + str(int(self.pressione[0] / np.sum(self.pressione) * 100))
            + "% \n"
        )
        f.write(txt7)

        txt8 = (
            "Canestri a SX:: "
            + str(self.canestri_sx)
            + "% \n"
        )

        f.write(txt8)

        txt9 = (
            "Canestri a DX:: "
            + str(self.canestri_dx)
            + "% \n"
        )

        f.write(txt9)

        return f

    def stat1(self, image, boxes_ball, boxes_team, team_numbers, fps, frame_id, tracker):
        if (len(boxes_ball) > 0) or (len(self.last_valid_ball) > 0):
            if len(boxes_ball) > 0:  # a new valid ball position from the det+tracker
                self.last_valid_ball = boxes_ball
            else:  # if the det+tracker doesn't find a ball use the last one position
                boxes_ball = self.last_valid_ball

            ball_players_distance = []

            for box in boxes_team:
                ball_players_distance.append(distance_boxes(box, boxes_ball[0]))

            #find all players near ball
            players_near_ball_idx = []
            sum_of_distances_min150 = 0
            for i in range(len(ball_players_distance)):
                if ball_players_distance[i] < 150:
                    players_near_ball_idx.append(i)
                    sum_of_distances_min150 += ball_players_distance[i]

            #it's a way to generate a score for each player near the ball that tell us how good could be this player
            player_near_ball_boxes_scores= []
            for i in players_near_ball_idx:
                score = 1 - (ball_players_distance[i]/sum_of_distances_min150)
                player_near_ball_boxes_scores.append((boxes_team[i], score))

            player_near_ball_boxes_scores.sort(reverse=True, key = lambda x: x[1])

            boxes = []
            scores = []
            for x in player_near_ball_boxes_scores:
                boxes.append(x[0])
                scores.append(x[1])
                break            
            
            tracker.track(image, boxes, scores, frame_id, boxes_ball[0])

            if tracker.player_with_ball_tracking is not None:
                image = draw_rect(image, tracker.player_with_ball_tracking, (0, 255, 0))

            
            if distance_boxes(tracker.player_with_ball_tracking, boxes_ball[0]) < 150:
                self.player_that_shot = tracker.player_with_ball_tracking

            txt = "-"
            player_index = np.argmin(ball_players_distance)
            if ball_players_distance[player_index] < 150:

                team_number = int(team_numbers[player_index])

                self.storia_possesso_palla.append(team_number)

                # the current team number is defined as the most recurrent number in the last 10 frame
                self.filtered_team_number = int(
                    np.median(self.storia_possesso_palla[-10:])
                )

                self.possesso_palla[self.filtered_team_number] = (
                    self.possesso_palla[self.filtered_team_number] + 1
                )

                self.player_with_ball_box = boxes_team[player_index]
                if self.player_with_ball_box is not None:
                    image = draw_rect(image, self.player_with_ball_box, (0, 0, 255))
                    circle_player(image, self.player_with_ball_box, 150)                

                if (self.filtered_team_number) == 0:
                    txt = "Arbitro"
                if (self.filtered_team_number) == 1:
                    txt = "Team 1"
                if (self.filtered_team_number) == 2:
                    txt = "Team 2"

                cv2.putText(
                    image,  
                    txt, 
                    (
                        int(boxes_team[player_index][0]),
                        int(boxes_team[player_index][1] - 60),
                    ),  # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                    1,  # font size
                    (40, 40, 40, 255),  # font color
                    3, # font stroke
                )  

            image = cv2.putText(
                image,
                "In possesso: {}".format(txt),
                (100, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (200, 200, 255),
                4,
            )

            image = cv2.putText(
                image,
                "Possesso palla (sec/tot)",
                (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (200, 200, 200),
                4,
            )

            elapsed = round(frame_id / fps, 1)

            image = cv2.putText(
                image,
                "   Arbitri: {}/{} s".format(
                    round(self.possesso_palla[0] / fps, 1), elapsed
                ),
                (100, 200 + (80 * 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )
            # Team 1
            image = cv2.putText(
                image,
                "   Team 1: {}/{} s".format(
                    round(self.possesso_palla[1] / fps, 1), elapsed
                ),
                (100, 200 + (80 * 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )
            # Team 2
            image = cv2.putText(
                image,
                "   Team 2: {}/{} s".format(
                    round(self.possesso_palla[2] / fps, 1), elapsed
                ),
                (100, 200 + (80 * 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )

        return image


    def stat2(self, image, boxes_ball):
        # ball position wrt half way line DX o SX
        if (len(boxes_ball) > 0) or (len(self.last_valid_ball) > 0):
            if len(boxes_ball) > 0:  # a new valid ball position from the det+tracker
                self.last_valid_ball = boxes_ball
            else:  # if the det+tracker doesn't find a ball use the last one position
                boxes_ball = self.last_valid_ball

            coor = boxes_ball[0]
            line_points_arr = np.asarray(self.line_points)
            p1 = [coor[0], coor[1]]

            # Return -1 left, 0 on line, +1 right
            ball_pos = ball_position(self.line_points[0], self.line_points[1], p1)

            distance_ball_center = abs((np.cross(line_points_arr[0] - p1, p1 - line_points_arr[1])) / np.linalg.norm(line_points_arr[0] - p1))
            self.history_distance_ball_center.append(distance_ball_center * ball_pos)

            self.ballDX = ball_pos == 1
            self.ballSX = ball_pos == -1

            if ball_pos == 1:
                self.ball_cumulative_position[0] = self.ball_cumulative_position[0] + 1
                image = cv2.putText(
                    image,
                    "Posizione palla: DX",
                    (100, 200 + (80 * 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (200, 200, 200),
                    4,
                )
            else:
                self.ball_cumulative_position[1] = self.ball_cumulative_position[1] + 1
                image = cv2.putText(
                    image,
                    "Posizione palla: SX",
                    (100, 200 + (80 * 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (200, 200, 200),
                    4,
                )
        else:
            image = cv2.putText(
                image,
                "Posizione palla: -",
                (100, 200 + (80 * 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (200, 200, 200),
                4,
            )

        return image


    def stat4(self, image, boxes_ball, boxes_team, team_numbers):
        if len(boxes_ball) > 0:
            ball_team1_distance = []
            ball_team2_distance = []

            # number_of_value = 10  #number of value for mean
            # distance_search = 250   #how far i must search for a crowded frame
            number_of_value = 10  # number of value for mean
            distance_search = 500  # how far i must search for a crowded frame

            for i, box in enumerate(boxes_team):
                if int(team_numbers[i]) == 1:
                    ball_team1_distance.append(distance_boxes(box, boxes_ball[0]))
                if int(team_numbers[i]) == 2:
                    ball_team2_distance.append(distance_boxes(box, boxes_ball[0]))

            self.history_mean_dist_team1.append(np.mean(ball_team1_distance))
            self.history_mean_dist_team2.append(np.mean(ball_team2_distance))

            # with the history of all the players near the ball find the contropiede
            if len(self.history_mean_dist_team1) > distance_search:
                # mean of players near ball in the last 5 frames
                mean_team1 = np.mean(self.history_mean_dist_team1[-number_of_value:])
                mean_team2 = np.mean(self.history_mean_dist_team2[-number_of_value:])

                if (mean_team1 + mean_team2) / 2 > 400:
                    # print("poco affollato:  mean:" +str((mean_team1+mean_team2)/2))
                    last_50_dist_team1 = self.history_mean_dist_team1[
                        -distance_search:-number_of_value
                    ]
                    last_50_dist_team2 = self.history_mean_dist_team2[
                        -distance_search:-number_of_value
                    ]

                    frame_crowded = 0
                    for i in range(len(last_50_dist_team1)):
                        # check if there were at least 6 players in a passed position around the ball
                        if ((last_50_dist_team1[i] + last_50_dist_team2[i]) / 2) < 300:
                            frame_crowded += 1

                    if frame_crowded > number_of_value and (
                        self.filtered_team_number != 2 or self.filtered_team_number != 1):
                       
                        # print("number of frame crowded befor a single player action:  "+str(frame_crowded))
                        # print(np.gradient(history_distance_ball_center[-80:]))
                        direction = np.mean(
                            np.gradient(self.history_distance_ball_center[-80:])
                        )

                        # print("Gradient: {}".format(direction))

                        (H, W) = image.shape[:2]

                        image = cv2.rectangle(
                            image,
                            (int((W / 2) - 200), 50),
                            (int((W / 2) + 200), 300),
                            (0, 0, 0),
                            -1,
                        )

                        cv2.putText(
                            image,
                            "Direction of Attack: ",
                            (int((W / 2) - 150), 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (200, 200, 200),
                            2,
                        )

                        if direction > 0:
                            cv2.arrowedLine(
                                image,
                                (int((W / 2)), 200),
                                (int((W / 2)) + 150, 200),
                                (200, 200, 200),
                                8,
                                tipLength=0.5,
                            )

                        if direction < 0:
                            cv2.arrowedLine(
                                image,
                                (int((W / 2)), 200),
                                (int((W / 2) - 150), 200),
                                (200, 200, 200),
                                8,
                                tipLength=0.5,
                            )
        return image

    def stat5(self, image):
        # searching crowded area
        number_of_value = 10  # number of value for mean

        (H, W) = image.shape[:2]

        mean_team1 = np.mean(self.history_mean_dist_team1[-number_of_value:])
        mean_team2 = np.mean(self.history_mean_dist_team2[-number_of_value:])
        image = cv2.putText(
            image,
            "Pressione avversaria",
            (100, 200 + (80 * 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (200, 200, 200),
            4,
        )

        if (mean_team1 + mean_team2) / 2 < 500:
            # print("affollato")
            if self.ballSX:
                # print("affollamento a SX")
                self.pressione[0] += 1

            if self.ballDX:
                # print("affollamento dx")
                self.pressione[1] += 1

        if np.sum(self.pressione) > 0:
            image = cv2.putText(
                image,
                "   Team 1: {}%".format(
                    str(int(self.pressione[1] / np.sum(self.pressione) * 100))
                ),
                (100, 200 + (80 * 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )
            image = cv2.putText(
                image,
                "   Team 2: {}%".format(
                    str(int(self.pressione[0] / np.sum(self.pressione) * 100))
                ),
                (100, 200 + (80 * 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )

        return image

    def stat6(self, image, image_blank, boxes_team, team_numbers, bg_sub=True, tracking_player_with_ball=True):
        def bg_update(current_frame, prev_bg, alpha):
            bg = alpha * current_frame + (1 - alpha) * prev_bg
            bg = np.uint8(bg)
            return bg
        
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

        #to highlight tracked player with ball
        #if self.player_that_shot != None:
        #    image = draw_rect(image, self.player_that_shot, (255, 0, 255))

        is_center = False

        if bg_sub: #center detection using background subtraction
            #preliminary tasks
            #rois extractions
            rux, ruy, ruw, ruh = self.right_ROI_up
            roi_can_dx_up = image_blank[ruy : ruy + ruh, rux : rux + ruw]
            rdx, rdy, rdw, rdh = self.right_ROI_down
            roi_can_dx_down = image_blank[rdy : rdy + rdh, rdx : rdx + rdw]

            lux, luy, luw, luh = self.left_ROI_up
            roi_can_sx_up = image_blank[luy : luy + luh, lux : lux + luw]
            ldx, ldy, ldw, ldh = self.left_ROI_down
            roi_can_sx_down = image_blank[ldy : ldy + ldh, ldx : ldx + ldw]

            #background subtractions
            diff_dx_up = cv2.absdiff(self.bg_right_up, roi_can_dx_up)
            diff_dx_down = cv2.absdiff(self.bg_right_down, roi_can_dx_down)
            diff_sx_up = cv2.absdiff(self.bg_left_up, roi_can_sx_up)
            diff_sx_down = cv2.absdiff(self.bg_left_down, roi_can_sx_down)

            #mask thresholding
            ret_dx_up, motion_mask_dx_up = cv2.threshold(diff_dx_up, self.threshold, self.maxval, cv2.THRESH_BINARY)
            ret_dx_down, motion_mask_dx_down = cv2.threshold(diff_dx_down, self.threshold, self.maxval, cv2.THRESH_BINARY)
            ret_sx_up, motion_mask_sx_up = cv2.threshold(diff_sx_up, self.threshold, self.maxval, cv2.THRESH_BINARY)
            ret_sx_down, motion_mask_sx_down = cv2.threshold(diff_sx_down, self.threshold, self.maxval, cv2.THRESH_BINARY)

            # Update background
            self.bg_right_up = bg_update(roi_can_dx_up, self.bg_right_up, alpha=self.alpha)
            self.bg_right_down = bg_update(roi_can_dx_down, self.bg_right_down, alpha=self.alpha)
            self.bg_left_up = bg_update(roi_can_sx_up, self.bg_left_up, alpha=self.alpha)
            self.bg_left_down = bg_update(roi_can_sx_down, self.bg_left_down, alpha=self.alpha)

            #convert motion_masks in grayscale
            motion_mask_dx_up = cv2.cvtColor(motion_mask_dx_up, cv2.COLOR_BGR2GRAY)
            motion_mask_dx_down = cv2.cvtColor(motion_mask_dx_down, cv2.COLOR_BGR2GRAY)
            motion_mask_sx_up = cv2.cvtColor(motion_mask_sx_up, cv2.COLOR_BGR2GRAY)
            motion_mask_sx_down = cv2.cvtColor(motion_mask_sx_down, cv2.COLOR_BGR2GRAY)

            # count non zero pixel into motion masks
            nzp_dx_up = cv2.countNonZero(motion_mask_dx_up)
            nzp_dx_down = cv2.countNonZero(motion_mask_dx_down)
            nzp_sx_up = cv2.countNonZero(motion_mask_sx_up)
            nzp_sx_down = cv2.countNonZero(motion_mask_sx_down)

            #look for a center
            if self.ballDX:
                image = draw_rect(image, self.right_ROI_down, (0, 255, 0))
                image = draw_rect(image, self.right_ROI_up, (0, 255, 0))
                #cv2.imshow('Motion mask dx_up', motion_mask_dx_up)
                if nzp_dx_up >= self.nzp_up_min:
                    #print("nzp_dx_up: ", nzp_dx_up)
                    self.delay_dx_up_val = self.delay_dx_up

                if self.nzp_down_max >= nzp_dx_down >= self.nzp_down_min and self.delay_dx_down_min_val <=0 and self.delay_dx_down_max_val <=0 and self.delay_dx_up_val > 0:
                    self.canestri_dx += 1
                    is_center = True
                    #print("nzp_dx_down: ", nzp_dx_down)
                    self.delay_dx_down_min_val = self.delay_dx_down_min
                elif nzp_dx_down >= self.nzp_down_max and self.delay_dx_down_min_val<=0 and self.delay_dx_up_val > 0:
                    self.delay_dx_down_max_val = self.delay_dx_down_max
                    #print("pubblicità rilevata a dx")

                if self.delay_dx_down_min_val > 0:
                    image = draw_rect(image, self.left_ROI_down, (0, 0, 255))
                    if self.player_with_ball_box is not None:
                        image = draw_rect(image, self.player_with_ball_box, (255, 255, 0)) #giocatore che ha tirato
                    if self.player_that_shot is not None:
                        image = draw_rect(image, self.player_that_shot, (255, 0, 255)) #giocatore che ha tirato secondo il tracking
                if self.delay_dx_up_val > 0:
                    image = draw_rect(image, self.right_ROI_up, (0, 0, 255))

            elif self.ballSX:
                image = draw_rect(image, self.left_ROI_down, (0, 255, 0))
                image = draw_rect(image, self.left_ROI_up, (0, 255, 0))
                #cv2.imshow('Motion mask sx_up', motion_mask_sx_up)
                #print("nzp_sx_up: ", nzp_sx_up)
                #print("nzp_sx_down: ", nzp_sx_down)
                if nzp_sx_up >= self.nzp_up_min:
                    #print("nzp_sx_up: ", nzp_sx_up)
                    self.delay_sx_up_val = self.delay_sx_up

                #cv2.imshow('Motion mask sx_down', motion_mask_sx_down)
                if self.nzp_down_max >= nzp_sx_down >= self.nzp_down_min and self.delay_sx_down_min_val <=0 and self.delay_sx_down_max_val <=0 and self.delay_sx_up_val > 0:
                    self.canestri_sx += 1
                    is_center = True
                    #print("nzp_sx_down: ", nzp_sx_down)
                    self.delay_sx_down_min_val = self.delay_sx_down_min
                elif nzp_sx_down >= self.nzp_down_max and self.delay_sx_down_min_val<=0 and self.delay_sx_up_val > 0:
                    self.delay_sx_down_max_val = self.delay_sx_down_max
                    #print("pubblicità rilevata a sx")

                if self.delay_sx_down_min_val > 0:
                    image = draw_rect(image, self.left_ROI_down, (0, 0, 255))
                    if self.player_with_ball_box is not None:
                        image = draw_rect(image, self.player_with_ball_box, (255, 255, 0)) #giocatore che ha tirato
                    if self.player_that_shot is not None:
                        image = draw_rect(image, self.player_that_shot, (255, 0, 255)) #giocatore che ha tirato secondo il tracking

                if self.delay_sx_up_val > 0:
                    image = draw_rect(image, self.left_ROI_up, (0, 0, 255))

        else: #center detection using ball tracking
            if len(self.last_valid_ball) > 0:  
                ball_box = self.last_valid_ball[0]
            else:
                ball_box = [0,0,0,0]

            #compute intersection between ball bounding box and basket bounding box
            overlap_right_up1 = get_overlap(ball_box, self.right_ROI_up)
            overlap_right_down1 = get_overlap(ball_box, self.right_ROI_down)
  
            overlap_left_up1 = get_overlap(ball_box, self.left_ROI_up)
            overlap_left_down1 = get_overlap(ball_box, self.left_ROI_down)

            overlap_right_up2 = get_overlap(self.right_ROI_up, ball_box)
            overlap_right_down2 = get_overlap(self.right_ROI_down, ball_box)

            overlap_left_up2 = get_overlap(self.left_ROI_up, ball_box)
            overlap_left_down2 = get_overlap(self.left_ROI_down, ball_box)

            #look for a center
            if self.ballDX:
                image = draw_rect(image, self.right_ROI_down, (0, 255, 0))
                image = draw_rect(image, self.right_ROI_up, (0, 255, 0))
                #print("self.delay_dx_up_val ", self.delay_dx_up_val)
                #print("self.delay_dx_down_min_val ", self.delay_dx_down_min_val)
                if overlap_right_up1 >= 0.40 or overlap_right_up2 >= 0.40:
                    #print("nzp_dx_up: ", nzp_dx_up)
                    self.delay_dx_up_val = self.delay_dx_up

                if (overlap_right_down1 >= 0.40 or overlap_right_down2 >= 0.40) and self.delay_dx_up_val > 0 and self.delay_dx_down_min_val < 0:
                    self.canestri_dx += 1
                    is_center = True
                    self.delay_dx_down_min_val = self.delay_dx_down_min

                if self.delay_dx_down_min_val > 0:
                    if self.right_ROI_down is not None:
                        image = draw_rect(image, self.right_ROI_down, (0, 0, 255))
                    if self.player_with_ball_box is not None:
                        image = draw_rect(image, self.player_with_ball_box, (255, 255, 0)) #giocatore che ha tirato
                    if self.player_that_shot is not None:
                        image = draw_rect(image, self.player_that_shot, (255, 0, 255)) #giocatore che ha tirato secondo il tracking

            elif self.ballSX:
                image = draw_rect(image, self.left_ROI_down, (0, 255, 0))
                image = draw_rect(image, self.left_ROI_up, (0, 255, 0))
                #print("self.delay_sx_up_val ", self.delay_sx_up_val)
                #print("self.delay_sx_down_min_val ", self.delay_sx_down_min_val)
                if overlap_left_up1 >= 0.40 or overlap_left_up2 >= 0.40:
                        #print("nzp_dx_up: ", nzp_dx_up)
                        self.delay_sx_up_val = self.delay_sx_up

                if (overlap_left_down1 >= 0.40 or overlap_left_down2 >= 0.40) and self.delay_sx_up_val > 0 and self.delay_sx_down_min_val < 0:
                    self.canestri_sx += 1
                    is_center = True
                    self.delay_sx_down_min_val = self.delay_sx_down_min

                if self.delay_sx_down_min_val > 0:
                    image = draw_rect(image, self.left_ROI_down, (0, 0, 255))
                    if self.player_with_ball_box is not None:
                        image = draw_rect(image, self.player_with_ball_box, (255, 255, 0)) #giocatore che ha tirato
                    if self.player_that_shot is not None:
                        image = draw_rect(image, self.player_that_shot, (255, 0, 255)) #giocatore che ha tirato secondo il tracking

        #to compute the history of player in the 3pt area, than used in 1pt
        team1_counter = 0
        team2_counter = 0
        for i, box in enumerate(boxes_team):
            #compreso chi sta tirando nell'area dei 3pt ci devono essee 3 giocatori per squadra
            x_box, y_box, w_box, h_box = box
            if self.mask_3pt[int(y_box+h_box), int(x_box)]==255 or self.mask_3pt[int(y_box+h_box), int(x_box+w_box)]==255:
                if int(team_numbers[i]) == 1:
                    team1_counter += 1
                if int(team_numbers[i]) == 2:
                    team2_counter += 1
        self.team1_counter_history.append(team1_counter)
        self.team2_counter_history.append(team2_counter)

        if tracking_player_with_ball:
            player_who_shot = self.player_that_shot
        else:
            player_who_shot = self.player_with_ball_box

        if player_who_shot is not None:
            x, y, w, h = player_who_shot#con tracking
            #controllo se c'è un giocatore nella lunetta del tiro libero 
            if image.shape[0] == self.mask_lunette.shape[0] and image.shape[1] == self.mask_lunette.shape[1]:
                if self.mask_lunette[int(y+h), int(x)]==255 or self.mask_lunette[int(y+h), int(x+w)]==255:
                    self.lunetta_history.append(True)
                else:
                    self.lunetta_history.append(False)

            if image.shape[0] == self.mask_3pt.shape[0] and image.shape[1] == self.mask_3pt.shape[1]:
                if self.mask_3pt[int(y+h), int(x)]==255 or self.mask_3pt[int(y+h), int(x+w)]==255:
                    self.area_3pt_history.append(True)
                else:
                    self.area_3pt_history.append(False)


        #try to understand which is the center value
        if is_center and player_who_shot is not None:            
            print("looking for the right value of the center")            
            x, y, w, h = player_who_shot
            #controllo se c'è un giocatore nella lunetta del tiro libero
            if ((len(self.lunetta_history) > 5) and (len(self.area_3pt_history) > 5) and 
            (len(self.team1_counter_history) >=5) and (len(self.team2_counter_history) >=5)):

                if self.lunetta_history[-5 :].count(True) >= self.lunetta_history[-5 :].count(False): 
                    mean1 = mean(self.team1_counter_history[-5 :])
                    mean2 = mean(self.team2_counter_history[-5 :])   
                    if ((2 < mean1 < 3) and (2 < mean2 < 3)):
                        print("tiro libero da 1 pt")
                        if self.ballDX:
                            self.punti_dx += 1
                        else:
                            self.punti_sx += 1
                    else:
                        if self.ballDX:
                            self.punti_dx += 2
                        else:
                            self.punti_sx += 2
                elif self.area_3pt_history[-5 :].count(True) >= self.area_3pt_history[-5 :].count(False):
                    print("canestro da 2 pt from history")
                    if self.ballDX:
                        self.punti_dx += 2
                    else:
                        self.punti_sx += 2
                else:
                    print("canestro da 3 pt from history")
                    if self.ballDX:
                        self.punti_dx += 3
                    else:
                        self.punti_sx += 3
            else:
                if self.area_3pt_history[-1]:
                    print("canestro da 2 pt")
                    if self.ballDX:
                        self.punti_dx += 2
                    else:
                        self.punti_sx += 2
                else:
                    print("canestro da 3 pt")
                    if self.ballDX:
                        self.punti_dx += 3
                    else:
                        self.punti_sx += 3

        image = cv2.putText(
            image,
            "Canestri segnati:",
            (3140, 120 + (80 * 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (200, 200, 200),
            4,
        )

        image = cv2.putText(
            image,
            "   Destra: {}".format(
                str(int(self.canestri_dx))
            ),
            (3140, 120 + (80 * 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
        )
        image = cv2.putText(
            image,
            "   Sinistra: {}".format(
                str(int(self.canestri_sx))
            ),
            (3140, 120 + (80 * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
        )

        image = cv2.putText(
            image,
            "Punti:",
            (3140, 120 + (80 * 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (200, 200, 200),
            4,
        )

        image = cv2.putText(
            image,
            "   Destra: {}".format(
                str(int(self.punti_dx))
            ),
            (3140, 120 + (80 * 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
        )
        image = cv2.putText(
            image,
            "   Sinistra: {}".format(
                str(int(self.punti_sx))
            ),
            (3140, 120 + (80 * 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
        )

        self.delay_dx_down_min_val -= 1
        self.delay_dx_down_max_val -= 1
        self.delay_dx_up_val -= 1

        self.delay_sx_down_min_val -= 1
        self.delay_sx_down_max_val -= 1
        self.delay_sx_up_val -= 1
        return image
 
    def run_stats(self, image, image_blank, boxes_ball, scores_ball, boxes_team, team_numbers, fps, frame_id, tracker, bg_sub, tracking_player_with_ball):
        # Draw stats windows
        image = cv2.rectangle(image, (50, 50), (700, 100 + (80 * 9)), (0, 0, 0), -1)
        image = cv2.rectangle(image, (3090, 50), (3790, 100 + (80 * 6)), (0, 0, 0), -1)

        # Draw pitch middle line
        image = cv2.line(
            image, self.line_points[0], self.line_points[1], (20, 20, 20), thickness=2
        )

        # Chiamata statistica 1
        #image = self.stat1(image, boxes_ball, boxes_team, team_numbers, fps, frame_id, tracker)

        # Chiamata statistica 2
        image = self.stat2(image, boxes_ball)

        # Chiamata statistica 4
        image = self.stat4(image, boxes_ball, boxes_team, team_numbers)

        # Chiamata statistica 5
        image = self.stat5(image)

        # Chiamata statistica 6
        image = self.stat6(image, image_blank, boxes_team, team_numbers, bg_sub, tracking_player_with_ball)

        return image


def run_all(video_path, ball_tracking_path, team_detection_path, out_txt_file, bg_sub, tracking_player_with_ball):
    ball_dict = get_dict(ball_tracking_path)
    team_dict = get_dict(team_detection_path)

    # Input video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(
        "m", "p", "4", "v"
    )  #output video format definition

    if bg_sub == True:
        point_detection = "bg_sub"
    else:
        point_detection = "ball_trk"

    if tracking_player_with_ball == True:
        trkPWB = "trkPWB_True"
    else:
        trkPWB = "trkPWB_False"

    out = cv2.VideoWriter(
        "./output_stats_video/" + args.video_name + "_" + args.ball_detector + args.ball_type + "_" + args.players_type + "_" + point_detection + "_" + trkPWB + ".mp4",
        fourcc,
        30.0,
        (int(video.get(3) / 2), int(video.get(4) / 2)),
        True,
    )  #output video properties definition

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    stat = Statistics()
    
    tracker = Tracker()

    ret, img = video.read()
    stat.initialize(img, 1)

    ret = True
    frame_id = 1
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while ret:
        ret, frame = video.read()

        if not ret:
            continue

        image = frame
        image_blank = frame.copy()

        #extraction of boxes of teams and ball from the given datasets
        #boxes are the coordinates of the ball/player, scores repr how good is the prediction, names is the frame id,
        #complete contains all the informations
        boxes_team, scores_team, names_team, team_numbers = (
            [[0, 0, 0, 0]],
            [[0]],
            [[0]],
            [[0]],
        )

        boxes_team, scores_team, names_team, complete_team, team_numbers = get_gt(
            frame, frame_id, team_dict
        )

        boxes_ball, scores_ball, names_ball, not_used = (
            [[0, 0, 0, 0]],
            [[0]],
            [[0]],
            [[0]],
        )
        boxes_ball, scores_ball, names_ball, complete_ball, not_used = get_gt(
            frame, frame_id, ball_dict
        )

        image = draw_players(image, boxes_team, team_numbers)
        if len(boxes_ball) > 0:
            coor = boxes_ball[0]  # return only one detection for the ball
            draw_rect(image, coor, (10, 255, 255))

        stat.run_stats(image, image_blank, boxes_ball, scores_ball, boxes_team, team_numbers, fps, frame_id, tracker, bg_sub, tracking_player_with_ball)

        image_to_show = cv2.resize(image, (1920, 1080))
        cv2.imshow("image", image_to_show)
        cv2.waitKey(1)

        (H, W) = image.shape[:2]
        i = cv2.resize(image, (int(W / 2), int(H / 2)))
        out.write(i)

        frame_id += 1

    out.release()

    # output file for statistics
    try:
        os.remove(out_txt_file)
    except:
        None
    f = open(out_txt_file, "a")

    stat.generate_file(f, frame_id)
    f.close()


############################################################
#  Main statistics
############################################################

if __name__ == "__main__":
    import argparse

    all_ok = True

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--video_name', required=True,
                        metavar="video name",
                        help='Video to apply the tracking on')
    parser.add_argument('--ball_type', required=True,
                        metavar="coordinates of the ball",
                        help='available values: detect, track, interp')
    parser.add_argument('--ball_detector', required=True,
                        metavar="how ball has been detected",
                        help='available values: yolo, rcnn')
    parser.add_argument('--players_type', required=True,
                        metavar="coordinates of the players",
                        help='available values: detect, track_new, track_sort, track_byte')
    parser.add_argument('--bg_sub', required=True,
                        metavar="type of point detection",
                        help='True: use background subtraction, False: use ball bounding box')
    parser.add_argument('--trkPWB', required=True,
                        metavar="True: track player with ball, False: use last player near ball",
                        help='True: track player with ball, False: use last player near ball')                    
    args = parser.parse_args()

    args.video = "./input_video/" + args.video_name + ".mp4"
    

    if args.ball_type == "detect":
        args.ball_track = "./det_tracking/ball_detection_" + args.ball_detector + "/" + args.video_name + ".txt"
    elif args.ball_type == "track":
        args.ball_track = "./det_tracking/ball_tracking_" + args.ball_detector + "/" + args.video_name + ".txt"
    elif args.ball_type == "interp":
        args.ball_track = "./det_tracking/ball_interpolation_" + args.ball_detector + "/" + args.video_name + ".txt"
    else:
        print("unvalid ball_type value")
        all_ok = False

    if args.players_type == "detect":
        args.det_teams = "./det_tracking/player_detection/" + args.video_name + ".txt"
    elif args.players_type == "track_new":
        args.det_teams = "./det_tracking/player_tracking/" + args.video_name + ".txt"
    elif args.players_type == "track_sort":
        args.det_teams = "./det_tracking/player_tracking_sort/" + args.video_name + ".txt"
    elif args.players_type == "track_byte":
        args.det_teams = "./det_tracking/player_tracking_byte/" + args.video_name + ".txt"
    else:
        print("unvalid players_type value")
        all_ok = False

    if args.bg_sub == "True":
        point_detection = "bg_sub"
        args.bg_sub = True
    else:
        point_detection = "ball_trk"
        args.bg_sub = False

    if args.trkPWB == "True":
        trkPWB = "trkPWB_True"
        args.trkPWB = True
    else:
        trkPWB = "trkPWB_False"
        args.trkPWB = False

    args.out_stats = "./det_tracking/stats/" + args.video_name + "_" + args.ball_detector + args.ball_type + "_" + args.players_type + "_" + point_detection + "_" + trkPWB + ".txt"

    if all_ok:
        run_all(args.video, args.ball_track, args.det_teams, args.out_stats, args.bg_sub, args.trkPWB)
