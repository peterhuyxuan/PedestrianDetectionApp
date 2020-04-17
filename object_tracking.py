
# Import python libraries
import cv2
import copy
import os
from human_detectors import Human_Detectors
from tracker import Tracker
from people_in_box import People_In_Box


def main():
    # video = cv2.VideoCapture('project.avi')

    # Initialise Human Detector
    human_detect = Human_Detectors()

    # Initialise Tracker
    tracker = Tracker(160, 30, 5, 100)

    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

    rootdir = "./sequence/"
    # # loop over the image paths
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdir, file)
            frame = cv2.imread(path)

        # # Run through video frames
        # while(True):
        #     ret, frame = video.read()

            # Detect and return centeroids of the objects in the frame
            centers = human_detect.Detect(frame)
            total_centers = len(centers)

            # Track centroids, if found
            if (total_centers > 0):
                # Track object using Kalman Filter
                tracker.Update(centers)
                count = len(tracker.tracks)

                # Start Task 2
                # Establishing coordinates and dimension of box
                x_coord = 300
                y_coord = 100
                width = 200
                height = 200

                # Declaring Constructor for the People in Box Tracker
                people_inside = People_In_Box(
                    frame, centers)

                # Return the count of the people inside the drawn box
                peopleInBox = people_inside.count_people(
                    x_coord, y_coord, x_coord + width, y_coord + height)

                # draw a rectangle
                cv2.rectangle(
                    frame, (x_coord, y_coord), (x_coord + width, y_coord + height), (255, 0, 0), 2)
                # End Task 2

                # Start Task 3 - Fix the function in people_in_box.py
                peopleInGroup, peopleAlone, frame = people_inside.count_people_in_group(
                    75)

                # drawing tracks with different colors
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Coordinated of predicted line
                            x_1 = int(tracker.tracks[i].trace[j][0][0])
                            y_1 = int(tracker.tracks[i].trace[j][1][0])
                            x_2 = int(tracker.tracks[i].trace[j+1][0][0])
                            y_2 = int(tracker.tracks[i].trace[j+1][1][0])
                            clr = tracker.tracks[i].track_id % 9
                            cv2.line(frame, (x_1, y_1), (x_2, y_2),
                                     track_colors[clr], 2)

                # Show the tracked video frame
                cv2.putText(frame, 'people detected: ' + str(count), (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'people in box: ' + str(peopleInBox), (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'people in groups: ' + str(peopleInGroup), (10, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'people alone: ' + str(peopleAlone), (10, 550),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('Tracking', frame)

                """
                For everyone running it, press "Spacebar" to go to the next frame
                """
                cv2.waitKey(
                    0)  # Comment out if you want to run like a video instead of frame by frame
                # print("total number people in the frame: ", count)

            key = cv2.waitKey(50) & 0xff
            # Escape key to exit
            if key == 27:
                break

    # video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
