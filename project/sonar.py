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


ranges = [(10, 50, 1), (51, 1000, 0)]
r_counter = [0] * len(ranges)
r_thresh = 10
try:
    while True:
        dist = distance()
        print(dist)
        for i, (rl, rh, v) in enumerate(ranges):
            if rl <= dist <= rh:
                r_counter[i] += 1
                for j in range(len(r_counter)):
                    if j != i:
                        r_counter[j] = 0

            for j, c in enumerate(r_counter):
                if c >= r_thresh:
                    VIB_pin.ChangeDutyCycle(int(100 * v))
                    time.sleep(2)
                    VIB_pin.ChangeDutyCycle(0)
                    r_counter = [0] * len(ranges)

except KeyboardInterrupt:
    VIB_pin.stop()
    GPIO.cleanup()


