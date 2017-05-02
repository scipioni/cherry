# USAGE
# python videostream_demo.py
# python videostream_demo.py --picamera 1

# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2
import collections

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

class CvTimer(object):
    def __init__(self):
	self.last = time.time()
	self.history = collections.deque(maxlen=100)

    @property
    def fps(self):
	now = time.time()
	self.history.append(1.0/(now - self.last))
	self.last = now
	
	return round(sum(self.history)/len(self.history))

# loop over the frames from the video stream
timer = CvTimer()
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=640)

	# draw the timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, str(timer.fps), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
