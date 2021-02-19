import cv2
import numpy as np

class Image:
    image = None
    kp = None
    des = None

    def __init__(self, image = None):
        self.image = image

    def setKpDesc(self, kp=None, des=None):
        self.kp = kp
        self.des = des

    def getDesc(self):
        return self.des

    def getKp(self):
        return self.kp
        

