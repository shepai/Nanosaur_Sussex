import RPi.GPIO as GPIO
import time

# Pin Definitions
pinA = 33  # 
pinB = 31  #

def main():
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(pinA, GPIO.IN)
    GPIO.setup(pinB, GPIO.IN)
    while True:
        print(GPIO.input(pinA),GPIO.input(pinB))
        time.sleep(0.5)
    GPIO.cleanup()

if __name__ == '__main__':
    main()
