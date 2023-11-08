from math import sqrt

import cv2

class Centroid:
    def __init__(self, id_centroid):
        self.centerPoints = []
        self.id_centroid = id_centroid

    def update(self, x, y):
        self.centerPoints.append((x, y))

    def last_pos(self):
        return self.centerPoints[-1]

    #é so para desenhar o centroid
    def draw(self, frame):
        id_text = f"ID {self.id_centroid}"
        center = self.last_pos()
        newFrame = cv2.putText(frame, id_text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        newFrame = cv2.circle(newFrame, center, 4, (255, 255, 255), -1)

        prev = self.centerPoints[0]
        for next in self.centerPoints[1:]:
            newFrame = cv2.line(newFrame, prev, next, (0, 0, 255), 2)#tracing
            prev = next
        return newFrame


