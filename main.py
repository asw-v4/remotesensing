import time
import cv2
import argparse
import math
import numpy as np
from imutils.video import FileVideoStream, FPS




class main:

    def __init__(self,args):
        initialisationPeriod = 10# increase with low SNR
        self.loadVideoStream(args)
        self.dim = None
        self.kernel = np.ones((3,3), np.uint8)
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

        if self.dim is None:
            scale_percent = 100 # percent of original size
            self.w = int(frame.shape[1] * scale_percent / 100)
            self.h = int(frame.shape[0] * scale_percent / 100)
            self.dim = (self.w, self.h)
        frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
        if frame is None:
            self.stop()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame/255
        frame.astype('float16')
        iCOR = self.correctImage(frame)
        # print(np.amax(frame), np.amin(frame))
        # print(np.amax(iCOR), np.amin(iCOR))
        return frame, iCOR

    def correctImage(self, frame):

        SAT = np.mean(frame)
        iCOR = frame-SAT

        return iCOR

    def initialiseFilters(self, initialisationPeriod):
        for self.n in range(initialisationPeriod):
            frame, iCOR = self.loadVideoFrame()
            self.initialiseBackgroundFilter(iCOR)
            self.initialiseVarianceFilter(frame, iCOR)
        self.iBG = self.rollingAvg_iCOR
        self.iVARsq = self.rollingAvg_iVARsq


        print("Initialised")

    def initialiseBackgroundFilter(self, iCOR):
        try:
            self.rollingAvg_iCOR = self.rollingAvg_iCOR + (iCOR - self.rollingAvg_iCOR)/self.n
        except:
            self.rollingAvg_iCOR = iCOR
            self.rollingAvg_iCOR.astype('float16')


    def initialiseVarianceFilter(self, frame, iCOR):
        try:
            self.rollingAvg_iVARsq = self.rollingAvg_iVARsq + (np.square(self.rollingAvg_iCOR - frame) - self.rollingAvg_iVARsq)/self.n
        except:
            self.rollingAvg_iVARsq = np.square(self.rollingAvg_iCOR-frame)
            self.rollingAvg_iVARsq.astype('float16')

    def updateThresholds(self, t1, t2, iVAR):

        T1 = t1*iVAR
        T2 = t2*iVAR
        return T1, T2

    def generateMasks(self, iCOR, iBG, THRESHOLD_ONE):
        iRES = iCOR - iBG
        absiRES = np.abs(iRES)
        # absolute value to invert negative values
        temp = THRESHOLD_ONE - absiRES
        temp = np.clip(temp, a_min=0, a_max=255)
        # Positive values are below THRESHOLD_ONE
        # Only update pixels below the threshold
        update_BG_mask = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY)[1]
        freeze_update_BG_mask = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY_INV)[1]
        # print(np.count_nonzero(update_BG_mask), np.count_nonzero(temp))
        # print(np.count_nonzero(freeze_update_BG_mask), temp.size - np.count_nonzero(temp))
        # Confirm correct values are being converted to 1 in each mask
        return update_BG_mask, freeze_update_BG_mask

    def predict_iBG(self, alpha, iCOR, iBG, update_BG_mask, freeze_BG_mask):
        update_BG_mask = update_BG_mask*(alpha*iBG + (1-alpha)*iCOR)
        freeze_BG_mask = freeze_BG_mask*iBG
        self.iBG = update_BG_mask + freeze_BG_mask
        self.iBG.astype('float16')
        # cv2.imshow("update", update_BG_mask)
        # cv2.imshow("freeze", freeze_BG_mask)
        cv2.imshow("self.iBG", abs(self.iBG))
        # print(np.amax(update_BG_mask), np.amin(update_BG_mask))

    def predict_iVAR(self, beta, iVARsq, iBG, iCOR, update_iVARsq_mask, freeze_iVARsq_mask):
        update_iVARsq_mask = update_iVARsq_mask*(beta*iVARsq + (1-beta)*np.square(iCOR - iBG))
        freeze_iVARsq_mask = freeze_iVARsq_mask*iVARsq
        self.iVARsq = update_iVARsq_mask + freeze_iVARsq_mask
        self.iVARsq.astype('float16')
        # print(np.amax(iVARsq), np.amin(iVARsq))
        cv2.imshow("iVARsq", self.iVARsq)

    def generateOutputMask(self, THRESHOLD_TWO, iBG, iCOR):
        absiRES = np.abs(iCOR - iBG)
        # cv2.imshow("Residual", absiRES) # normed to 1
        # absolute value to invert negative values
        # print(np.amax(absiRES), np.amin(absiRES))
        # print(np.amax(THRESHOLD_TWO), np.amin(THRESHOLD_TWO))
        temp = absiRES - THRESHOLD_TWO
        output = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY)[1]

        # MOVING_mask = np.clip(absiRES, a_min=THRESHOLD_TWO, a_max=1)
        # MOVING_mask = (MOVING_mask - THRESHOLD_TWO)*255
        # print(np.count_nonzero(temp),np.count_nonzero(MOVING_mask) )
        return output

    def display_mask(self, output, overlay=False):

        cv2.imshow("MASK", output)
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.stop()

    def checkFrozen(self, iBG, iCOR):
        try:
            diff = abs(self.lastBG - iBG)
            self.lastBG = iBG
        except:
            self.lastBG = iBG
            diff = np.zeros((self.h,self.w))
        # print(np.amax(iCOR), np.amax(self.iBG))
        no_diff_mask = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY_INV)[1]
        # print(np.amax(diff*no_diff_mask), np.amin(diff*no_diff_mask)) # always 0
        # all with no diff = 1
        self.freeze_mask = np.multiply(self.freeze_mask,no_diff_mask) + no_diff_mask
        # resets values to 0 if no_diff is 0, else + 1

        # only values = max_freeze will = 1
        frozen_1 = cv2.threshold(self.freeze_mask, self.max_freeze, 1, cv2.THRESH_BINARY)[1]
        inv_to_reset = cv2.threshold(self.freeze_mask, self.max_freeze, 1, cv2.THRESH_BINARY_INV)[1]
        # sets frozen values to 0, and non froezn to 1
        self.iBG = self.iBG*inv_to_reset + iCOR*frozen_1
        self.iVARsq = self.iVARsq*inv_to_reset + np.square(iCOR-self.iBG)*frozen_1
        self.freeze_mask = self.freeze_mask*inv_to_reset
        # print(np.count_nonzero(self.freeze_mask))
        # resets frozen values

    def run(self, initialisationPeriod, args):
        t1 = float(args.thr1)
        t2 = float(args.thr2)

        assert t1 < t2

        alpha = 0.9999
        beta = 0.999
        self.initialiseFilters(initialisationPeriod)
        self.max_freeze = int(args.warmup)
        self.freeze_mask = np.zeros((self.h, self.w))

        while self.stream.more() is True:

            frame, iCOR = self.loadVideoFrame()
            cv2.imshow("INPUT", frame)
            #frame and iCOR are now one frame ahead of iBG and iVAR
            #self.checkFrozen(self.iBG, iCOR, frame)
            self.iVAR = np.sqrt(self.iVARsq)
            print(np.amax(self.iVARsq))
            THRESHOLD_ONE, THRESHOLD_TWO = self.updateThresholds(t1, t2, self.iVAR)

            update, freeze = self.generateMasks(iCOR, self.iBG, THRESHOLD_ONE)
            # Generates masks and residual image
            self.predict_iBG(alpha, iCOR, self.iBG, update, freeze)
            # self.iBG is now inline with iCOR and frame
            self.predict_iVAR(beta, self.iVARsq, self.iBG, iCOR, update, freeze)
            # self.iVARsq is now inline with iCOR, frame and iBG
            self.checkFrozen(self.iBG, iCOR)
            output = self.generateOutputMask(THRESHOLD_TWO, self.iBG, iCOR)
            # removed_singles = cv2.erode(output, self.kernel, iterations=2)
            # filled_mask = cv2.dilate(removed_singles, self.kernel, iterations=1)
            # output = cv2.blur(filled_mask, (3,3))
            self.display_mask(output)





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-v', '--video', help='path to IR Video')
    argparser.add_argument('-t1', '--thr1', help='Estimation Threshold')
    argparser.add_argument('-t2', '--thr2', help='Output Sensitivty')
    argparser.add_argument('-w', '--warmup', help='How long to tolerate frozen pixels')

    args = argparser.parse_args()
    main(args)
