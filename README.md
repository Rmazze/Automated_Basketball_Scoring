# Automatic statistics generator on basketballs video 

Computer Vision Project A.A. 2021/2022

To develop this project we started from two previous existing works. \
The first one is completely based on Mask R-CNN detector that can be found [here](https://github.com/simoberny/basket_tracking).\
The second one exploit YOLO detector for the detection of the ball and can be found [here](https://github.com/MatteoDalponte/Basketball_statistics).

We uploaded the code and all the necessary material in this [drive folder](https://drive.google.com/drive/u/1/folders/1fUSfXSfIG1YqJCk6XCSigHIsqn-Har1N).

The code of our project is also available in a GitLab repository available at this [link](https://gitlab.com/R_mazze/cv_automatic_basketball_scoring.git).

# Installation: 

To utilize the codebase first you need to install the following versions in order to be able to use last CUDA version for faster computation:

```
python = 3.8.10
opencv = 4.4.0
tensorflow = 2.11.0
```

1. Install the maskrcnn requirement. For doing so:
     - First install the required libraries that can be found at 
     ```mask_rcnn_UPG_x_playerDetection/requirements.txt```
     - After run `python3 setup.py` st the same location of before
2. Download the weights files from [here](https://drive.google.com/drive/folders/1lfB7KuQnpw1sAYpMv0qFP9bu1NM45A2s?usp=share_link) and put them in a folder called weights in the directory of the project.
4. Download the test video files from [here](https://drive.google.com/drive/folders/1lfB7KuQnpw1sAYpMv0qFP9bu1NM45A2s?usp=share_link) and put it in a folder called input_video in the directory of the project
5. Create a folder `output_stats_video`
6. Create a folder `det_tracking` containing the sub-folders
   - `ball_detection_rcnn`
   - `ball_detection_yolo`
   - `ball_interpolation_rcnn`
   - `ball_interpolation_yolo`
   - `ball_tracking_byte`
   - `ball_tracking_rcnn`
   - `ball_tracking_yolo`
   - `player_detection`
   - `player_tracking`
   - `player_tracking_byte`
   - `player_tracking_sort`
   - `stats`

# How to run:

The framework is structured in different parts where:

1. **Detetection of the ball with yolo** 
     - COMMAND: `python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball832x832.cfg -w weights/yolov3_ball_5000_832x832_augmented.weights -v input_video/video_name.mp4 -s -out_txt det_tracking/video_name_aug1.txt` 
     - INPUT: 
          - YOLO configurations
          - YOLO weights
          - video_file: e.g. undistorted_canestro4.mp4
     - OUT: 
          - file txt in MOT format with the detection positions
          - output_video

2. **Ball tracking** 
     - COMMAND: `python custom_tracking_v2.py --video_name video_name --ball_detector ball_detection_method` \
     - INPUT:
       - video_name: e.g. undistorted_canestro4
       - ball_detection_method: available values are `yolo` , for detection done with YOLO, and `rcnn` , for the detection done with Mask R-CNN.
     - OUT: 
       - file txt with tracking
       - output_video

3. **Ball interpolation** 
     - COMMAND: `python interpolation.py --video_name video_name --ball_detector ball_detection_method`
     - INPUT:
       - video_name: e.g. undistorted_canestro4
       - ball_detection_method: available values are `yolo` , for detection done with YOLO, and `rcnn` , for the detection done with Mask R-CNN.
     - OUT: 
       - file txt with interpolated tracking
       - output_video

4. **Players detection** 
     - COMMAND: `cd mask_rcnn_UPG_x_playerDetection1/samples/player_detection`\
          `python player_detection1.py --weights ../../../weights/mask_rcnn_coco.h5 --video ../../../input_video/video_name.mp4 -d --command detect`
     - INPUT:
       - Mask R-CNN weights trained on COCO
       - video_file: e.g. undistorted_canestro4.mp4
     - OUT: 
       - file txt with player divided by teams
       - output_video

5. **Players tracking with customized tracker** 
   - COMMAND: `python player_tracking_new.py --video_name video_name`
   - INPUT:
       - video_name: e.g. undistorted_canestro4
   - OUT: 
     - file txt with tracked player divided by teams
     - output_video

6. **Players tracking with sort tracker** 
   - COMMAND: `python player_tracking_sort.py --video_name video_name`
   - INPUT:
       - video_name: e.g. undistorted_canestro4
   - OUT: 
     - file txt with tracked player divided by teams
     - output_video

7. **Players tracking with byte tracker** 
   - COMMAND: `python player_tracking_byte.py --video_name video_name`
   - INPUT:
       - video_name: e.g. undistorted_canestro4
   - OUT: 
     - file txt with tracked player divided by teams
     - output_video

8. **statistics generation**
     - COMMAND: `python generate_stats.py --video_name video_name --ball_type ball_coordinates_type --ball_detector ball_detection_method --players_type players_coordinates_type --bg_sub shoot_detection_flag --trkPWB player_with_ball_tracking_flag`
     - INPUT: 
       - video_name: e.g. undistorted_canestro4
       - ball_coordinates_type: available values are `detect`  to use coordinets obtained with the detection, `track`  to use coordinates obtained after tracking, `interp`  to use coordinates obtained after interpolation. Suggested vale `interp` .
       - ball_detection_method: available values are `yolo` , for detection done with YOLO, and `rcnn` , for the detection done with Mask R-CNN. Suggested vale "YOLO".
       - players_coordinates_type: available values are `detect`  to use the coordinates obtained after the detection, `track_new`  to use coordinates obtained after tracking with the customized tracker, `track_sort`  to use coordinates obtained after tracking with sort tracker, `track_byte`  to use coordinates obtained after tracking with byte tracker. Suggested vale `track_byte` .
       - shoot_detection_flag: available values are `True`  to use background subtraction for shot detection, `False`  to use the method based on ball coordinates. Suggested vale `True` .
       - player_with_ball_tracking_flag: available values are `True`  or `False`  to enable or disable the tracking of the player with the ball. Suggested vale `True` .
     - OUT: 
       - stats.txt
       -  out_video

# Long run basketball shots results:

We benchmarked our shots detection architecture against a video of a real match of basketball. The video represents the full first time of the match for a length of almost 25 minutes.

**N.B:** The system is taylored to work with videos in undistorted format.

Here a table reporting the results of the test.

|Methodology            | TP | FP | FN | Precion | Accuracy | F1 score |
|-----------------------|----|----|----|---------|----------|----------|
|Ball Tracking          | 3  | 1  | 21 | 0.75    | 0.13     | 0.21     |
|Back Ground Subtraction| 16 | 14 | 8  | 0.53    | 0.67     | 0.59     |