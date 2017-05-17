import RPi.GPIO as GPIO
import time

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

pin=11 # or GPIO17
GPIO.setup(pin, GPIO.OUT)

GPIO.output(pin, GPIO.LOW)
while True:
    print("H")
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(1.0)
    GPIO.output(pin, GPIO.LOW)
    print("L")
    time.sleep(1.0)
