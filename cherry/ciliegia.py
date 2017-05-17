# -*- coding:utf-8 -*-

import numpy as np
import cv2
import time
from cherry.lib import cvutil

def draw_text(img, text, y=75, x=10, color=(255,255,0)):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

class Ciliegie:
    def __init__(self):
        self.ciliegie = []

    def update(self, ciliegie):
        """
        aggancia le nuove ciliegie a quelle presenti
        """
        pass

class Ciliegia:
    _area_min = 50 # normalizzata e in millesimi
    _aspect_ratio_level = 0.4 # percentuale di scostamento massima dal quadrato

    def __init__(self, contour, h_parent, w_parent, taratura=28.0):
        """
            la ciliegia ai bordi dell'immagine ha un calibro un 10% inferiore rispetto al centro
            a causa della distanza dall'obbiettivo della camera
        """
        self.contour = contour
        self.h_parent = h_parent
        self.mirino = None
        self.x = 0
        self.y = 0

        self.vs = []
        self.last = time.time()
        self.fired = False

        self.rect = cv2.minAreaRect(contour)
        xy,wh,theta = self.rect
        self.w,self.h = wh
        self.x,self.y = map(int,xy) # centro della ciliegia
        self.area_norm = 1000*self.w*self.h/(h_parent*w_parent) # normalizziamo alla dimensione del parente e in millesimi
        self.y_norm = 1.0*self.y/h_parent
        self.calibro = 0
        if self.w > 0:
            self.aspect_ratio = 1.0*self.h/self.w # il quadrato ha 1
        else:
            self.aspect_ratio = 0

    def set_mirino(self, mirino):
        self.mirino = mirino
        self.calibro = round(10.0*self.mirino.taratura*max(self.w, self.h)/self.h_parent)/10

        if self.calibro > mirino.calibro-2: # quello che non è passato dal calibro precedente
            distance = 100.0*(self.y-mirino.y)/self.h_parent
            #print distance
            if 0 <= distance <= 2:
                mirino.fire()
                print "FIRED"

    def draw(self, img):
        cv2.circle(img, (self.x, self.y), 4, cvutil.yellow, 2)
        if cvutil.major >= 3:
            box = cv2.boxPoints(self.rect) # cv2.boxPoints(rect) for OpenCV 3.x
        else:
            box = cv2.cv.BoxPoints(self.rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], -1, cvutil.green, 1)
        cv2.putText(img, str(self.calibro), (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cvutil.yellow)

    def is_valid(self):
        if not self.w or not self.h:
            return False
        if self.area_norm < self._area_min:
            #print("BAD area %s" % self.area_norm)
            return False
        if abs(1-self.aspect_ratio) > self._aspect_ratio_level:
            #print("BAD aspect ratio %s" % self.aspect_ratio)
            return False
        if self.w*self.h < 10: # troppo piccolo
            return False
        return True

    '''
    def update(self, img, hull, x, y, area, y_min):
        posizione = float(y-y_min)/y_min
        if posizione > 0.8:
            print("nuova ciliegia...")
            self.fired = False
        now = time.time()
        v = float(self.y-y)/(now-self.last)
        self.x = x
        self.y = y
        self.last = now
        if 6 < v < 1000:
            self.vs.append(v)
        draw_text(img, area, y, x-20)
        draw_text(img, int(self.get_v()), y-50, x, color=(0,0,255))
        cv2.drawContours(img, [hull], -1, (0,255,0), 2)
        if not self.fired and posizione < 0.6 and v > 6:
            self.fire()

    def get_v(self):
        if not self.vs:
            return 0
        return sum(self.vs)/len(self.vs)

    def fire(self):
        self.fired = True
        v = self.get_v()
        if not v:
            return
        print(" FIRE v=%d vs=%s" % (v, self.vs))
        self.vs = []
    '''
