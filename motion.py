import cv2
import numpy as np
from image import Image

class MotionPath:
    numberOfImages = None
    images = None
    orb = None
    matcher = None

    coords = None

    def __init__(self):
        self.numberOfImages = 0
        self.images = []
        self.orb = cv2.ORB_create()
        self.coords = []
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def matchPoints(self, img1, img2):
        des1 = img1.getDesc()
        des2 = img2.getDesc()

        matches = self.matcher.match(des1, des2)
        return matches
        
    def getFundamentalMatrix(self, mathces):
        pass

    def readNext(self, image):
        kp, des = self.orb.detectAndCompute(image, None)
        img = Image(image)
        img.setKpDesc(kp, des)
        self.images.append(img)

        if len(self.images) > 1:
            img1 = self.images[len(self.images) - 2]
            img2 = self.images[len(self.images) - 1]
            matches = self.matchPoints(img1, img2)
            F = self.getFundamentalMatrix(matches)

            



