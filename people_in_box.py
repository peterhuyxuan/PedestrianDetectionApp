import cv2


class People_In_Box(object):
    # Class to detect people in a box
    def __init__(self, frame, centers):
        self.people_in_box = 0
        self.frame = frame
        self.centers = centers

    # Function to count the people in the box
    def count_people(self, x1, y1, x2, y2):
        people_in_box = 0

        # Extracting the coordinates from the centroids
        for x, y in self.centers:
            cX = x[0]
            cY = y[0]
            # Increasing count of people inside the box
            if cX > x1 and cX < x2 and cY > y1 and cY < y2:
                people_in_box += 1

        # Returning the count
        return people_in_box

    # Function to count the people in the box
    def count_people_in_group(self, threshold):
        """
        Function for Task 3 to fix
        """
        # Establish box with threshold to include other nearby people
        people_in_group = 0
        people_alone = 0

        # Extracting the coordinates from the centroids
        for x, y in self.centers:
            x1 = x[0] - threshold
            y1 = y[0] - threshold
            x2 = x[0] + threshold
            y2 = y[0] + threshold

            # Calculating people in group within threshold
            people_in_group = self.count_people(x1, y1, x2, y2)
            # Differentiating single pedestrian from groups of people
            if people_in_group == 1:
                people_in_group = 0
                people_alone += 1
            # Draws the box if a group (more than 1 people) is detected
            elif people_in_group > 1:
                # Attempt to keep count value consistent throught but unsure if it is working
                people_in_group = self.count_people(x1, y1, x2, y2)
                # draw a rectangle
                cv2.rectangle(
                    self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Returning the count of people witin a group as well as the frame with the drawn box
        return people_in_group, people_alone, self.frame
