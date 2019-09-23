import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
TRIG = 4
ECHO = 20
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

VIB = 21
GPIO.setup(VIB, GPIO.OUT)
VIB_pwm = GPIO.PWM(VIB, 100) # set Frequece to 1KHz
VIB_pwm.start(0)             # Start PWM output, Duty Cycle = 0

button = 25
GPIO.setup(button, GPIO.IN)

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


def line(x):
    y = 133.8 - x
    if y > 100:
        y = 100
    if y < 0:
        y = 0
    return y

try:
    while True:
        ACTIVE = 1
        thresh = 5
        analyze_cnt = 0
        while True:
            analyze_val = GPIO.input(button)
            if analyze_val == ACTIVE:
                analyze_cnt += 1
            else:
                analyze_cnt = 0

            if analyze_cnt >= thresh:
                analyze_cnt = 0
                dist = round(distance(), 2)
                # print('Current Distance :', dist)

                print('Distance is :', dist)
                print('Intensity is :', line(dist))
                print()
                print()
                print()

                VIB_pwm.ChangeDutyCycle(line(dist))
                time.sleep(2)
                VIB_pwm.ChangeDutyCycle(0)
                time.sleep(0.5)

except KeyboardInterrupt:
    VIB_pwm.stop()
    GPIO.cleanup()


