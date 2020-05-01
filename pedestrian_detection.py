import cv2
import os
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict, deque
from tensorflow_detection_api import DetectorAPI
from centroid_tracking import CentroidTracker
from people_in_box import PeopleInBox

# Initialise

img_path = './input_img'
out_path = './output_img'
model_path = 'frozen_inference_graph.pb'

writeVideo_flag = True
out_video = 'final_pedestrian_output_vid.avi'

# counter = 0

odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7

np.random.seed(123)
color = np.random.choice(range(256), size=3)
color = [int(c) for c in color]

# define box for task 2- to be changed to stdin input
bxx = 150
bxy = 100
bxw = 450
bxh = 350

fps = 10


def get_file_names(img_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        for file in filenames:
            files.append(file)

    return files


def save_to_video():
    print("Saving to video...")
    out_files = get_file_names(out_path)
    out_files.sort()
    img = cv2.imread(os.path.join(out_path, out_files[0]))
    height, width, layers = img.shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    print("Outputting video...")
    for f in out_files:
        img = cv2.imread(os.path.join(out_path, f))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)

    print("Video saved to " + out_video)
    out.release()
    cv2.destroyAllWindows()


def main():
    filenames = get_file_names(img_path)
    filenames.sort()
    start = time.time()
    people_array = []
    peopleingroups = []
    group_array = []
    frame_array = []
    ct = CentroidTracker()

    # Default dictionary of format {id:[[centroid[0],centroid[1]],..], id[...]}
    pts_dict = defaultdict(lambda: deque(maxlen=10))

    for f in filenames:

        if not f.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue

        print(f)
        img = cv2.imread(os.path.join(img_path, f))

        # Track and Detect Pedestrians
        # Results of pretrained frcnn detection.
        boxes, scores, classes, num = odapi.processFrame(img)

        rects = []
        rects_centers = []
        n = 0

        for i in range(len(boxes)):
            # keep bounding boxes o20f human (Class 1) with probability above threshold
            # and height greater than 40
            if classes[i] == 1 and scores[i] > threshold and (boxes[i][2] - boxes[i][0]) >= 35:

                box = boxes[i]
                # x1 = box[1]
                # y1 = box[0]
                # x2 = box[3]
                # y2 = box[2]

                cv2.rectangle(img, (box[1], box[0]),
                              (box[3], box[2]), (color), 2)
                rects.append(box)

                cx = int((box[1]+box[3])/2)
                cy = int((box[0]+box[2])/2)
                # cv2.circle(img,(cx, cy),5,(color),-1)
                rects_centers.append([cx, cy])

                # number of pedestrian detected
                n += 1

        objects = ct.update(rects)
        centers = []
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[1] - 10, centroid[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 0), 2)
            cv2.circle(img, (centroid[1], centroid[0]), 4, (225, 225, 0), -1)
            centers.append([centroid[1], centroid[0]])
            pts_dict[objectID].append((centroid[1], centroid[0]))
            img = odapi.drawTrail(pts_dict, img)

        print('Finish detection and tracking...')

        # Return the count of the people inside the drawn box

        # Initialise Constructor for the People in Box Tracker
        people_inside = PeopleInBox(img, rects_centers, rects)

        peopleInBox, _ = people_inside.count_people_in_box(
            bxx, bxy, bxx+bxw, bxy+bxh)

        # draw user defined bounding rectangle
        cv2.rectangle(img, (bxx, bxy), (bxx+bxw, bxy+bxh),
                      (255, 255, 204), 1)

        print('Finish counting in the box...')

        # Track groups of pedestrians

        # select centroid distance of 40 as group detection threshold
        groups = people_inside.detect_group(40)
        peopleInGroup, peopleAlone, group_boxs = people_inside.count_people_in_group(
            groups)

        # draw group bounding boxes
        if len(group_boxs) > 0:
            for box in group_boxs:
                cv2.rectangle(img, (box[0], box[1]),
                              (box[2], box[3]), (51, 225, 255), 2)

        # Show the count on frame
        text_color = [150, 255, 0]
        (text_width, text_height), baseline = cv2.getTextSize(
            'People detected: ', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, 'People detected: ' + str(n),
                    (10, 15+(text_height+5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
        cv2.putText(img, 'People in box: ' + str(peopleInBox),
                    (10, 15+(text_height+5)*2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
        cv2.putText(img, 'Groups detected: ' + str(len(group_boxs)),
                    (10, 15+(text_height+5)*3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
        cv2.putText(img, 'People in groups: ' + str(peopleInGroup),
                    (10, 15+(text_height+5)*4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
        cv2.putText(img, 'People alone: ' + str(peopleAlone),
                    (10, 15+(text_height+5)*5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
        people_array.append(n)
        group_array.append(len(group_boxs))
        peopleingroups.append(peopleInGroup)
        frame_array.append(img)

        cv2.imwrite(os.path.join(out_path, f),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print('End of', f)
        print('\n')

    cv2.destroyAllWindows()

    end = time.time()
    print('Run time:', (end-start))

    if writeVideo_flag:
        save_to_video()

    height, width, layers = frame_array[0].shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = 0
    fps = 10

    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    for i in range(len(frame_array)):
        img = cv2.cvtColor(frame_array[i], cv2.COLOR_BGR2RGB)
        out.write(img)

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    main()
