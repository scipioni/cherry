# -*- coding:utf-8 -*-

import numpy as np
import cv2
import time
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import argparse

from cherry.lib import cvutil
from cherry.ciliegia import Ciliegia


class Mirino:
    def __init__(self, y=0.5, delta=0.20):
        self.ciliegia = Ciliegia()

    def update(self, img, y=0.5, delta=0.2):
        """
            y_min e y_max hanno lo zero nella parte bassa dell'immagine
        """
        height, width = img.shape[:2]

        self.y_max = int(height*y+height*delta)
        self.y_min = int(height*y-height*delta)
        cv2.rectangle(img, (0,height-self.y_max), (width,height-self.y_min), cvutil.yellow, 2)
        #return (cy_min, cy_max)

class Detector:
    def __init__(self, show=False, filename=''):
        self.show = show
        self.timer = cvutil.CvTimer()


        self.cap = cv2.VideoCapture(filename or 0)
        #print("framerate %s" % self.cap.get(cv2.cv.CV_CAP_PROP_FPS))

        self.window = cvutil.Window('original')
        #self.hue_min = self.window.add_trackbar('hue_min', 1, 180, default=2)
        self.hue_max = self.window.add_trackbar('hue_max', 1, 180, default=25)
        self.saturation_min = self.window.add_trackbar('saturation_min', 1, 255, default=25)
        #self.saturation_max = self.window.add_trackbar('saturation_max', 1, 255, default=255)
        self.value_min = self.window.add_trackbar('value_min', 0, 255, default=0)
        #self.value_max = self.window.add_trackbar('value_max', 1, 255, default=255)
        self.w = self.window.add_trackbar('w', 0.2, 0.8, step=0.01, default=0.4)
        self.w_offset = self.window.add_trackbar('w_offset', 0.0, 0.2, step=0.01, default=0.08)
        self.mirino1_y = self.window.add_trackbar('mirino1_y', 0.1, 0.8, step=0.01, default=0.35)
        self.mirino2_y = self.window.add_trackbar('mirino2_y', 0.1, 0.8, step=0.01, default=0.80)
        self.mirino_h = self.window.add_trackbar('mirino_h', 0.1, 0.8, step=0.01, default=0.12)

        self.window_result = cvutil.Window('result') #, size=(160,600))

        self.mirino1 = Mirino()
        self.mirino2 = Mirino()

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

    def filter_color(self, img):
        #denoise = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        #small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #cv2.imshow('hsv', hsv)
        #self.show_hsv(hsv)
        mask1 = cv2.inRange(hsv,
            (0, self.saturation_min.getInt(), self.value_min.getInt()),
            (self.hue_max.getInt(), 255, 255),
            )
        result = cv2.bitwise_and(img, img, mask=mask1)
        return result

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

    def crop(self, frame):
        slice=self.w.get()
        delta=self.w_offset.get()
        h, w, depth = frame.shape
        x1 = int(w*(slice+delta))
        x2 = int(w*(1-slice+delta))
        result = frame[0:h, x1:x2].copy()
        cv2.rectangle(frame, (x1-2,0), (x2+2,h), cvutil.blu, 2)
        return result

    def main(self):
        pause = False
        i=0
        while(True):
            if not pause:
                ret, original = self.cap.read()
            if original is None:
                print("no frame")
                break
            #cv2.imshow('original', frame)

            original_show = original.copy()
            frame = self.crop(original_show)
            self.mirino1.update(original_show, y=self.mirino1_y.get(), delta=self.mirino_h.get())
            self.mirino2.update(original_show, y=self.mirino2_y.get(), delta=self.mirino_h.get())

            frame = self.filter_color(frame)

            #self.calculate(result, mask)

            fps = self.timer.fps
            if i%10:
                print("fps=%s" % fps)
                #cv2.putText(img, "fps=%.0f" % fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cvutil.blue)
            if self.show:
                self.window.show(original_show)
                self.window_result.show(frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                pause = not pause



            time.sleep(0.1)

        self.cap.release()
        cv2.destroyAllWindows()

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action='store_true', help="show camera")
    ap.add_argument("--file", default="")
    args = vars(ap.parse_args())
    detector = Detector(show=args['show'], filename=args['file'])
    detector.main()
