from collections import OrderedDict
import numpy as np


class PeopleInBox(object):

    # Class to detect people in a box
    def __init__(self, frame, centers, rects):
        # self.people_in_box = 0
        self.frame = frame
        self.centers = centers
        self.rects = rects

    # Function to count the people in the box
    def count_people_in_box(self, x1, y1, x2, y2):
        people_in_box = 0
        # pts = []
        pid = 0
        pids = []

        # Extracting the coordinates from the centroids
        for x, y in self.centers:
            cX = x
            cY = y

            # Increasing count of people inside the box
            if cX > x1 and cX < x2 and cY > y1 and cY < y2:
                people_in_box += 1
                # pts.append([cX, cY])
                pids.append(pid)

            pid += 1

        # Returning the count, centroid index and centroid coordinates
        return people_in_box, pids

    # Function to merge group with common pids
    def merge(self, lists, results=None):

        if results is None:
            results = []

        if not lists:
            return results

        first = lists[0]
        merged = []
        output = []

        for li in lists[1:]:
            for i in first:
                if i in li:
                    merged = merged + li
                    break
            else:
                output.append(li)

        merged = merged + first
        results.append(list(set(merged)))

        return self.merge(output, results)

    # Function to detect group in the frame
    def detect_group(self, threshold):
        """
        detect groups of people based on the centroid distance threshold
        output: groups of indices of centers
        """
        groups = []
        # gid = 0
        # pid = 0

        # Extracting the coordinates from the centroids
        for x, y in self.centers:
            x1 = x - threshold
            y1 = y - threshold
            x2 = x + threshold
            y2 = y + threshold

            # Extracting group information
            count, pid_in_group = self.count_people_in_box(x1, y1, x2, y2)
            if pid_in_group not in groups:
                # print(pid_in_group)
                groups.append(pid_in_group)

        # Returning groups of indices of centers
        return self.merge(groups)

    # Function to count the people in group and construct group bounding box
    def count_people_in_group(self, groups):

        people_in_group = 0
        people_alone = 0
        group_boxs = []
        margin = 5

        if len(groups) > 0:

            for cluster in groups:

                # Calculate bounding box coordinates of groups in the frame
                if len(cluster) > 1:
                    people_in_group += len(cluster)
                    boxs = [self.rects[i] for i in cluster]
                    boxs = np.stack(boxs, axis=1)

                    x1 = max(min(boxs[1])-margin, 1)
                    y1 = max(min(boxs[0])-margin, 1)
                    x2 = min(max(boxs[3])+margin, self.frame.shape[1] - 1)
                    y2 = min(max(boxs[2])+margin, self.frame.shape[0] - 1)

                    group_boxs.append([x1, y1, x2, y2])

                else:
                    people_alone += 1

        return people_in_group, people_alone, group_boxs
