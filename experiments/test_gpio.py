import RPi.GPIO as GPIO
import time

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

channel=11
GPIO.setup(channel, GPIO.OUT)

GPIO.output(channel, GPIO.LOW)
while True:
    print("H")
    GPIO.output(channel, GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(channel, GPIO.LOW)
    print("L")
    time.sleep(3)
