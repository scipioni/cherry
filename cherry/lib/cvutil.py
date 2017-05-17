# -*- coding:utf-8 -*-

import cv2
import numpy as np
import collections
import time

red=(0,0,255)
green=(0,255,0)
blu=(255,0,0)
yellow=(0,255,255)
white=(255,255,255)
black=(0,0,0)

(major, minor) = map(int, cv2.__version__.split(".")[:2])

_windows = []

class Track:
    def nothing(*args):
        pass

    def __init__(self, name, lower=2.0, upper=8.0, step=1.0, default=2.5, window='img1'):
        self.name = name
        self.step = step
        self.window = window
        cv2.createTrackbar(name, window, int(lower/step), int(upper/step), self.nothing)
        cv2.setTrackbarPos(name, window, int(default/step))

    def get(self):
        return cv2.getTrackbarPos(self.name, self.window)*self.step

    def getInt(self):
        return int(self.get())


class Window:
    w=600
    h=400

    def __init__(self, name='img1', size=None):
        self.name = name
        self.x = self.y = 1
        #cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        #return
        if not size:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL if size else cv2.WINDOW_AUTOSIZE)
        if size:
            cv2.resizeWindow(name, *size)
        if _windows:
            y = _windows[-1].y
            x = _windows[-1].x + self.w
            if x > 1920:
                x = 1
                y += self.h
            self.move(x,y)
        else:
            self.move(x=1,y=1)

        _windows.append(self)

    def move(self, x=1, y=1):
        self.x = x
        self.y = y
        cv2.moveWindow(self.name, x, y)

    def add_trackbar(self, name, lower=1, upper=10, step=1, default=5):
        return Track(name, lower=lower, upper=upper, step=step, default=default, window=self.name)

    def show(self, image):
        #smaller = cv2.resize(image, (self.w, self.h))
        #cv2.imshow(self.name, smaller)
        cv2.imshow(self.name, image)

def overlay(background, image, x=0, y=0):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rows,cols,channels = image_color.shape
    #background[0:rows, 0:cols] = image_color
    try:
        background[y:y+rows, x:x+cols] = image_color
    except:
        pass
    return background

def crop(image, size=None, slice=0.33):
    h, w, depth = image.shape

    # crop
    out = image[int(h*slice):int(h*(1-slice)), int(w*slice):int(w*(1-slice))]
    ho, wo, deptho = out.shape

    # resize
    #print("crop from %dx%d -> %dx%d -> %dx%d" % (w,h, wo,ho, self.__size[0],self.__size[1]))
    if size:
        return cv2.resize(out, size)
    return out

class CvTimer(object):
    def __init__(self):
	self.last = time.time()
	self.history = collections.deque(maxlen=10)

    @property
    def fps(self):
        now = time.time()
        self.history.append(1.0/(now - self.last))
        self.last = now
        return round(sum(self.history)/len(self.history))

    @property
    def avg_fps(self):
        return sum(self.l_fps_history) / float(self.fps_len)
