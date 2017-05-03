# -*- coding:utf-8 -*-

import numpy as np
import cv2
import time
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import argparse

from cherry.lib import cvutil
from cherry.ciliegia import Ciliegia

class Detector:
    def __init__(self, show=False, filename=''):
        self.show = show
        self.timer = cvutil.CvTimer()
        self.ciliegia = Ciliegia()

        self.cap = cv2.VideoCapture(filename or 0)
        #print("framerate %s" % self.cap.get(cv2.cv.CV_CAP_PROP_FPS))

        self.window = cvutil.Window('result')
        self.hue_min = self.window.add_trackbar('hue_min', 1, 180, default=2)
        self.hue_max = self.window.add_trackbar('hue_max', 1, 180, default=22)
        self.saturation_min = self.window.add_trackbar('saturation_min', 1, 255, default=110)
        self.saturation_max = self.window.add_trackbar('saturation_max', 1, 255, default=255)
        self.value_min = self.window.add_trackbar('value_min', 1, 255, default=120)
        self.value_max = self.window.add_trackbar('value_max', 1, 255, default=255)

    def erode(self, img, kernel=5):
        kernel_ = np.ones((kernel,kernel),np.uint8)
        return cv2.erode(img, kernel_, iterations = 1)

    def dilate(self, img, kernel=5):
        kernel_ = np.ones((kernel,kernel),np.uint8)
        return cv2.dilate(img, kernel_, iterations = 1)

    def show_hsv(self, hsv):
        hue, saturation, value = cv2.split(hsv)
        cv2.imshow('hue', hue)
        cv2.imshow('saturation', saturation)
        cv2.imshow('value', value)

    def detect_object(self, img):
        #denoise = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        #small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #cv2.imshow('hsv', hsv)
        #self.show_hsv(hsv)
        mask1 = cv2.inRange(hsv,
            (self.hue_min.getInt(), self.saturation_min.getInt(), self.value_min.getInt()),
            (self.hue_max.getInt(), self.saturation_max.getInt(), self.value_max.getInt()),
            )

        return mask1

    def calculate(self, img_out, img_mask):
        if cvutil.major >= 3:
            buff, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours,hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        cy_min, cy_max = self.mirino(img_out)

        hierarchy = hierarchy[0]

        #y=110
        for component in zip(contours, hierarchy):
            currentContour = component[0]
            currentHierarchy = component[1]
            if currentHierarchy[3] >= 0:
                continue

            rect = cv2.minAreaRect(currentContour)
            xy,wh,theta = rect
            w,h = wh
            if w*h < 10: # troppo piccolo
                return
            x,y = xy # centro della ciliegia

            # calcoliamo

            # visualizziamo il rect
            box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            cv2.drawContours(img_out, [box], -1, cvutil.yellow, 2)

            area = cv2.contourArea(currentContour)
            area = area/100.0

            if area > 10:
                hull = cv2.convexHull(currentContour)
                area = cv2.contourArea(hull)
                area = round(area/100)
                #hull = cv2.fitEllipse(currentContour)
                M = cv2.moments(hull)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                if cy < cy_min or cy > cy_max:
                    continue

                #color = ((0,0,255),(0,255,0))[cx < 200]
                #if cx < 200:
                #    area = round(area*1.2)
                self.ciliegia.update(img_out, hull, cx, cy, area, cy_min)
                #y += 50


    def mirino(self, img, delta=0.20):
        height, width = img.shape[:2]
        cy_max = int(height/2+height*delta)
        cy_min = int(height/2-height*delta)
        cv2.rectangle(img, (0,cy_min), (width,cy_max), (255,0,255), 2)
        return (cy_min, cy_max)

    def process(self, frame):
        # crop
        h, w, depth = frame.shape
        slice=0.40
        delta=0.08
        frame = frame[0:h, int(w*(slice+delta)):int(w*(1-slice+delta))]


        mask = self.detect_object(frame)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        self.calculate(result, mask)
        fps = self.timer.fps
        print("fps=%s" % fps)
        #cv2.putText(img, "fps=%.0f" % fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cvutil.blue)
        if self.show:
            cv2.imshow('result', result)

    def capture(self):
        pause = False
        while(True):
            if not pause:
                ret, frame = self.cap.read()
            if frame is None:
                print("no frame")
                break
            #cv2.imshow('original', frame)
            self.process(frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                pause = not pause

        self.cap.release()
        cv2.destroyAllWindows()

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action='store_true', help="show camera")
    ap.add_argument("--file", default="")
    args = vars(ap.parse_args())
    detector = Detector(show=args['show'], filename=args['file'])
    detector.capture()
