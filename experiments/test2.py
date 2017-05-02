"""

white balance: 2500 (no auto)



"""


import numpy as np
import cv2
import time
import collections
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse



def draw_text(img, text, y=75, x=10, color=(255,255,0)):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

#def circular_counter(max):
#    """helper function that creates an eternal counter till a max value"""
#    x = 0
#    while True:
#        if x == max:
#            x = 0
#        x += 1
#        yield x

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


def nothing(x):
    pass

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






class Detector:
    def __init__(self, raspberry=True, show=False):
        self.show = show
	self.timer = CvTimer()
        self.raspberry = raspberry
        self.whitebalance = 1.9
        self.ciliegia = Ciliegia()

        if raspberry:
            self.camera = PiCamera()
            self.camera.resolution = (640, 480)
            self.camera.framerate = 30
            self.camera.shutter_speed = self.camera.exposure_speed
            print("exposure_speed=%s" % self.camera.exposure_speed)
            self.camera.exposure_mode = 'off'
            g = self.camera.awb_gains
            self.camera.awb_mode = 'off'
            print("gains=", g)
            self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        else:
            self.cap = cv2.VideoCapture(0)
        #print("framerate %s" % self.cap.get(cv2.cv.CV_CAP_PROP_FPS))

        red = 0 # cv2.cvtColor(np.uint8([[[255,0,0]]]), cv2.COLOR_BGR2HSV)[0]
        self.ranges = [
                [ np.array([115,180,0]), np.array([180,255,255])],
#                [ np.array([0,115,110]), np.array([20,255,234])],
                ]
        cv2.namedWindow('image')
        range_ = self.ranges[0]
        cv2.createTrackbar('hue_min','image',range_[0][0],180,nothing)
        cv2.createTrackbar('hue_max','image',range_[1][0],180,nothing)
        cv2.createTrackbar('saturation_min','image',range_[0][1],255,nothing)
        cv2.createTrackbar('saturation_max','image',range_[1][1],255,nothing)
        cv2.createTrackbar('value_min','image',range_[0][2],255,nothing)
        cv2.createTrackbar('value_max','image',range_[1][2],255,nothing)
        cv2.createTrackbar('white_balance','image', int(round(self.whitebalance*10)),80,nothing)


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

        range_ = self.ranges[0]
        range_[0][0] = cv2.getTrackbarPos('hue_min','image')
        range_[1][0] = cv2.getTrackbarPos('hue_max','image')
        range_[0][1] = cv2.getTrackbarPos('saturation_min','image')
        range_[1][1] = cv2.getTrackbarPos('saturation_max','image')
        range_[0][2] = cv2.getTrackbarPos('value_min','image')
        range_[1][2] = cv2.getTrackbarPos('value_max','image')

        mask1 = cv2.inRange(hsv, *range_)
        #mask2 = cv2.inRange(hsv, *self.ranges[1])
        #mask1 = self.dilate(self.erode(mask1))
        #return mask1#+mask2

        return mask1
    

    def calculate(self, img_out, img_mask):
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

    def capture(self):
        if self.raspberry:
            self.capture_raspberry()
        else:
            self.capture_pc()


    def process(self, frame):
        mask = self.detect_object(frame)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        self.calculate(result, mask)
        fps = self.timer.fps
        print("fps=%s" % fps)
        draw_text(result, "fps=%.0f" % fps)
        if self.show:
            cv2.imshow('result', result)

    def capture_raspberry(self):
        for frameraw in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            whitebalance = cv2.getTrackbarPos('white_balance','image')/10.0
            self.camera.awb_gains = whitebalance
            self.process(frameraw.array)

            self.rawCapture.truncate(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def capture_pc(self):
        while(True):
            ret, frame = self.cap.read()
            #cv2.imshow('original', frame)
            self.process(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action='store_true', help="show camera")
    args = vars(ap.parse_args())
    detector = Detector(raspberry=False, show=args['show'])
    detector.capture()

