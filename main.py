import time
import cv2
import argparse
import math
import numpy as np
from imutils.video import FileVideoStream



class _main_:

    def __init__(self, args):
        self.stream = FileVideoStream(args.video).start()
        frame = self.stream.read()
        h,w,*_ = frame.shape
        print(frame.shape)
        self.SAT = 1/(h*w)
        self.initializationFrames = 75
        self.frameCount = 0

        self.alpha = 0.98
        self.initBG = np.zeros((h,w))

        self.beta = 0.98
        self.initVar = np.zeros((h,w))

        while self.stream.more():
            frame = self.stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #removes useless colour channels
            iCOR = self.driftRemoval(frame)
            if self.frameCount < self.initializationFrames:
                self.initializeBackgroundEstimation(iCOR)
                self.initializeVarienceEstimation(iCOR)
            elif self.frameCount == self.initializationFrames:
                self.initVar = self.initVar - (self.initBG*self.initBG)
                self.initVar = [math.sqrt(pix) for pix in self.initVar]

            #Remove average pixel values

            self.display(frame)




    def initializeBackgroundEstimation(self, iCOR):
        self.initBG = (self.initBG+iCOR) / 2



    def initializeVarienceEstimation(self, iCOR):
        sqiCOR = [pix * pix for pix in iCOR]
        self.initVar = (self.initVar+sqiCOR) / 2




    def driftRemoval(self, frame):
        iCOR = [pix - self.SAT for pix in frame]
        return iCOR

    def stop(self):
        self.stream.stop()
        cv2.destroyAllWindows()

    def display(self, frame):
        cv2.imshow("Output", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.stop()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-v', '--video', help='path to Thermal/IR Video')

    args = argparser.parse_args()
    _main_(args)
