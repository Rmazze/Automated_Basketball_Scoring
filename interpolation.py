import math

from utility.stat_utility import *


def save_mot(dic, txt):
    f = open(txt, "w")

    i = 0
    while (i in dic): 
        track, score, _, _ = get_gt(i, dic)

        if track != []:
            f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(i, track[0][0], track[0][1], track[0][2], track[0][3], score[0]))

        i+=1

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



def interpolation(detection_path):

    #Convert file detection to dictionary
    gt_dict = get_dict(detection_path)

    inter_frame = 0
    i = 0
    nt = 0

    while (i in gt_dict): 
        boxes, scores, names, complete = get_gt(i, gt_dict)

        if boxes == []:
            nt+=1

        #Check if there is hole of max 50 frames
        if boxes != [] and nt > 0 and nt < 50:
            print("NT: {}, index ora: {}".format(nt, i))

            before_index = i-(nt+1)

            if before_index > 0: 
                before = gt_dict[before_index][0]['coords']
                after = gt_dict[i][0]['coords']
                
                space_x = (after[0] - before[0])/(nt+1)
                space_y = (after[1] - before[1])/(nt+1)

                eucl = math.sqrt((after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2)

                if eucl > 800: 
                    nt = 0
                    continue

                print("Space x: {}, before: {}, after: {}".format(space_x, before[0], after[0]))

                for k in range(nt):
                    new_pos_x = before[0] + space_x * (k + 1)
                    new_pos_y = before[1] + space_y * (k + 1)

                    gt_dict[before_index + (k + 1)].append({'coords': [new_pos_x, new_pos_y, before[2], before[3]],'conf':1, 'ids':-1})
                    inter_frame+=1

            nt = 0

        i+=1

    return gt_dict
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Interpolation of the ball tracking')
    parser.add_argument('--video_name', required=True,
                        metavar="video name",
                        help='Video to apply the tracking on')
    parser.add_argument('--ball_detector', required=True,
                        metavar="how ball has been detected",
                        help='available values: yolo, rcnn')
    args = parser.parse_args()

    args.video = "./input_video/" + args.video_name + ".mp4"
    args.det = "./det_tracking/ball_tracking_" + args.ball_detector + "/" + args.video_name + ".txt"
    args.out_interpolation = "./det_tracking/ball_interpolation_" + args.ball_detector + "/" + args.video_name + ".txt"

    dict_finale = interpolation(args.det)
    #save_to_video(dict_finale, args.video, 'output/det_track_inter.mp4')
    save_mot(dict_finale, args.out_interpolation)
