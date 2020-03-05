import time
import cv2
import argparse
import math
import numpy as np
from imutils.video import FileVideoStream, FPS




class main:

    def __init__(self,args):
        initialisationPeriod = 150 # increase with low SNR
        self.loadVideoStream(args)
        self.run(initialisationPeriod, args)

    def stop(self):
        self.stream.stop()
        cv2.destroyAllWindows()

    def loadVideoStream(self, args):
        # Defined globally within the class,
        # as it is regularly called from nested functions
        self.stream = FileVideoStream(args.video).start()

    def loadVideoFrame(self):
        frame = self.stream.read()
        if frame is None:
            self.stop()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        iCOR = self.correctImage(frame)
        # print(np.amax(frame), np.amin(frame))
        # print(np.amax(iCOR), np.amin(iCOR))
        return frame, iCOR

    def correctImage(self, frame):
        try:
            SAT = self.invNPix * frame
            iCOR = frame - SAT
        except:
            self.nR, self.nC, *_ = frame.shape
            self.invNPix = 1/(self.nR*self.nC)
            SAT = self.invNPix * frame
            iCOR = frame - SAT
        # print(np.amax(SAT), np.amin(SAT))
        return iCOR

    def initialiseFilters(self, initialisationPeriod):
        for n in range(initialisationPeriod):
            frame, iCOR = self.loadVideoFrame()
            self.initialiseBackgroundFilter(iCOR)
            self.initialiseVarianceFilter(frame, iCOR)
        self.iBG = self.rollingAvg_iCOR
        self.iVARsq = self.rollingAvg_iVARsq
        print("Initialised")

    def initialiseBackgroundFilter(self, iCOR):
        try:
            self.rollingAvg_iCOR = np.average([self.rollingAvg_iCOR, iCOR], axis=0)
        except:
            self.rollingAvg_iCOR = iCOR

    def initialiseVarianceFilter(self, frame, iCOR):
        try:
            self.rollingAvg_frame = np.average([self.rollingAvg_frame, frame], axis=0)
            self.holdingVal = np.square((self.rollingAvg_iCOR - self.rollingAvg_frame))
            self.rollingAvg_iVARsq = np.average([self.rollingAvg_iVARsq, self.holdingVal], axis=0)
        except:
            self.rollingAvg_frame = frame
            self.holdingVal = np.square((self.rollingAvg_iCOR - self.rollingAvg_frame))
            self.rollingAvg_iVARsq = self.holdingVal

    def updateThresholds(self, t1, t2, iVAR):
        T1 = t1*iVAR
        T2 = t2*iVAR
        return T1, T2

    def generateMasks(self, iCOR, iBG, THRESHOLD_ONE):
        self.iRES = iCOR - iBG
        absiRES = abs(self.iRES)
        # absolute value to invert negative values
        temp = THRESHOLD_ONE - absiRES
        temp = np.clip(temp, a_min=0, a_max=99999999)
        # Positive values are below THRESHOLD_ONE
        # Only update pixels below the threshold
        update_BG_mask = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY)[1]
        freeze_update_BG_mask = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY_INV)[1]
        print(np.count_nonzero(update_BG_mask), np.count_nonzero(temp))
        print(np.count_nonzero(freeze_update_BG_mask), temp.size - np.count_nonzero(temp))
        # Confirm correct values are being converted to 1 in each mask
        return update_BG_mask, freeze_update_BG_mask

    def predict_iBG(self, alpha, iCOR, iBG, update_BG_mask, freeze_BG_mask):
        update_BG_mask = update_BG_mask*alpha*iBG + (1-alpha)*iCOR
        freeze_BG_mask = iBG
        self.iBG = update_BG_mask + freeze_BG_mask

    def predict_iVAR(self, beta, iVARsq, iBG, iCOR, update_BG_mask, freeze_BG_mask):
        update_BG_mask = update_BG_mask*beta*iVARsq + (1-beta)*(np.square(iCOR - iBG))
        freeze_BG_mask = iVARsq
        self.iVARsq = update_BG_mask + freeze_BG_mask

    def generateOutputMask(self, THRESHOLD_TWO):
        absiRES = abs(self.iRES)
        # absolute value to invert negative values
        temp = absiRES - THRESHOLD_TWO
        temp = np.clip(temp, a_min=0, a_max=99999999)

        MOVING_mask = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV)[1]
        return MOVING_mask

    def display_mask(self, MOVING):
        cv2.imshow("MASK", MOVING)
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.stop()

    def run(self, initialisationPeriod, args):
        t1 = float(args.thr1)
        t2 = float(args.thr2)
        assert t1 < t2

        alpha = 0.98
        beta = 0.98
        self.initialiseFilters(initialisationPeriod)

        while self.stream.more() is True:
            iVAR = np.sqrt(self.iVARsq)
            THRESHOLD_ONE, THRESHOLD_TWO = self.updateThresholds(t1, t2, iVAR)

            frame, iCOR = self.loadVideoFrame()
            #frame and iCOR are now one frame ahead of iBG and iVAR

            update, freeze = self.generateMasks(iCOR, self.iBG, THRESHOLD_ONE)
            # Generates masks and residual image
            self.predict_iBG(alpha, self.iBG, iCOR, update, freeze)
            # self.iBG is now inline with iCOR and frame
            self.predict_iVAR(beta, self.iVARsq, self.iBG, iCOR, update, freeze)
            # self.iVARsq is now inline with iCOR, frame and iBG
            # print(np.count_nonzero(self.iRES), self.iRES.size - np.count_nonzero(self.iRES), self.iRES.size )
            # Counts moving pixels (middle value)
            MOVING = self.generateOutputMask(THRESHOLD_TWO)
            self.display_mask(MOVING)





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-v', '--video', help='path to IR Video')
    argparser.add_argument('-t1', '--thr1', help='Estimation Threshold')
    argparser.add_argument('-t2', '--thr2', help='Output Sensitivty')


    args = argparser.parse_args()
    main(args)
