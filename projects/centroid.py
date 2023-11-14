import cv2
import time
from math import sqrt

class Centroid:
    def __init__(self, id_centroid):
        self.centerPoints = []
        self.id_centroid = id_centroid
        self.last_update_time = time.time()
        self.inside = False

    def update(self, x, y):
        self.centerPoints.append((x, y))
        self.last_update_time = time.time()

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

class CentroidTracker:
    def __init__(self, max_norma=30, max_inactive_time=2):
        self.centroids = {} #dicionário p guardar centroids
        self.max_norma = max_norma
        self.max_inactive_time = max_inactive_time
        self.id_counter = 0

    def within_valid_range(self, c1, c2):
        c1x, c1y = c1
        c2x, c2y = c2
        # cacula a norma
        norma = sqrt(((c2x - c1x) ** 2) + ((c2y - c1y) ** 2))
        # print("Norma: ", norma)
        if norma <= self.max_norma:
            return True
        return False

    def update(self, x, y):
        current_time = time.time()

        # Remove inactive centroids
        to_remove = [id_ for id_, centroid in self.centroids.items() if current_time - centroid.last_update_time > self.max_inactive_time]
        for id_ in to_remove:
            del self.centroids[id_]

        for id_, centroid in self.centroids.items():
            if self.within_valid_range((x, y), centroid.last_pos()):
                centroid.update(x, y)
                return centroid

        aux = Centroid(self.id_counter)
        aux.update(x, y)
        self.centroids[self.id_counter] = aux
        self.id_counter += 1
        return aux
