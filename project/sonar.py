import time

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
TRIG = 18
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

VIB = 40
VIB_pin = GPIO.PWM(VIB, 1000)     # set Frequece to 1KHz
VIB_pin.start(0)                     # Start PWM output, Duty Cycle = 0


def distance():
    def distance():
        GPIO.output(TRIG, True)

        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        StartTime = time.time()
        StopTime = time.time()

        while GPIO.input(ECHO) == 0:
            StartTime = time.time()

        while GPIO.input(ECHO) == 1:
            StopTime = time.time()

        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2

        return distance


range_low = 0
range_high = 50
counter = 0
try:
    while True:
        dist = distance()
        if range_low <= dist <= range_high:
            counter += 1
        if counter >= 10:
            VIB_pin.ChangeDutyCycle(80)
            time.sleep(2)
            VIB_pin.ChangeDutyCycle(0)
            counter = 0
except KeyboardInterrupt:
    VIB_pin.stop()
    GPIO.cleanup()


