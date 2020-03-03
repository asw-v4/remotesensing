import time
import cv2
import argparse
import math
import numpy as np
from imutils.video import FileVideoStream



class _main_:

    def __init__(self, args):
        self.stream = FileVideoStream(args.video).start()
        self.t1 = float(args.thr1)
        self.t2 = float(args.thr2)

        frame = self.stream.read()
        h,w,*_ = frame.shape
        self.SAT = 1/(h*w)
        self.initializationFrames = 75
        self.frameCount = 0

        self.alpha = 0.98
        self.initBG = np.zeros((h,w))

        self.beta = 0.98
        self.initVar = np.zeros((h,w))

        assert 0 < self.alpha <= 1
        assert 0 < self.beta <= 1
        assert 0 < self.t1 <= 1
        assert 0 < self.t2 <= 1

        self.initializeFilters()
        self.run()


    def initializeFilters(self):
        for n in range(self.initializationFrames+1):
            frame = self.stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #removes useless colour channels
            iCOR = self.driftRemoval(frame)
            if n < self.initializationFrames:
                self.initializeBackgroundEstimation(iCOR)
                self.initializeVarienceEstimation(iCOR, frame)
            elif n == self.initializationFrames:
                self.iVAR = np.sqrt(self.initVar)
                self.iBG = self.initBG


    def run(self):
        while self.stream.more():
            frame = self.stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #removes useless colour channels
            self.iCOR = self.driftRemoval(frame)
            self.iRES = self.iCOR - self.iBG
            
            stdiCOR = np.std(self.iCOR, axis=(0,1))
            THRESHOLD_ONE = self.t1*stdiCOR
            THRESHOLD_TWO = self.t2*stdiCOR

            if self.iCOR - self.iBG < THRESHOLD_ONE:
                self.iBG = self.alpha*self.iBG + (1-self.alpha)*self.iCOR







            self.display(frame)






    #def estimateNextBackground(self, )

    def initializeBackgroundEstimation(self, iCOR):
        self.initBG = (self.initBG+iCOR) / 2


    def initializeVarienceEstimation(self, iCOR, frame):
        val = (iCOR - frame)**2
        self.initVar = (self.initVar+val) / 2


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
    argparser.add_argument('-t1', '--thr1', help='Estimation Threshold')
    argparser.add_argument('-t2', '--thr2', help='Output Sensitivty')


    args = argparser.parse_args()
    _main_(args)
