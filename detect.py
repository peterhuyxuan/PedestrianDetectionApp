from centroidtracker import CentroidTracker
import numpy as np
import os
from imutils.object_detection import non_max_suppression
import cv2

rootdir = "./sequence/"

# initialise Centroid Tracker
ct = CentroidTracker()
(H, W) = (None, None)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# # loop over the image paths
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        img = cv2.imread(path)
        original = img.copy()
        cv2.imshow("Post-NMS", img)
        # detect people in the image
        rectangle, weights = hog.detectMultiScale(img, winStride=(4, 4),
                                                  padding=(8, 8), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rectangle:
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rectangle = np.array([[x, y, x + w, y + h]
                              for (x, y, w, h) in rectangle])
        # print(rectangle)
        pick = non_max_suppression(rectangle, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (x1, y1, x2, y2) in pick:
            cv2.rectangle(img, (x1, y1), (x2, x2), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        filename = path[path.rfind("/") + 1:]
        print("[File] {}: {} original boxes, {} after suppression".format(
            filename, len(rectangle), len(pick)))

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(pick)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # show the output images
        # cv2.imshow("Pre-NMS", original)
        cv2.imshow("Post-NMS", img)
        cv2.waitKey(0)
