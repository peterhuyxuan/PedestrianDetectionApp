import cv2


class People_In_Box(object):
    # Class to detect people in a box
    def __init__(self, frame, centers, x, y, width, height):
        self.people_in_box = 0
        self.frame = frame
        self.Height = frame.shape[1]
        self.Width = frame.shape[0]
        self.centers = centers
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # Function to count the people in the box
    def count_people(self):
        # Setting coordinates
        x1 = self.x
        y1 = self.y
        x2 = x1 + self.width
        y2 = y1 + self.height

        # draw a rectangle
        cv2.rectangle(
            self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Extracting the coordinates from the centroids
        for x, y in self.centers:
            cX = x[0]
            cY = y[0]
            # Increasing count of people inside the box
            if cX > x1 and cX < x2 and cY > y1 and cY < y2:
                self.people_in_box += 1

        # Returning the count and the frame with the drawn box
        return self.people_in_box, self.frame
