# -*- coding:utf-8 -*-

import cv2
import time

def draw_text(img, text, y=75, x=10, color=(255,255,0)):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)


class Ciliegia:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.area = 0
        self.vs = []
        self.last = time.time()
        self.fired = False

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
