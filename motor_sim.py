#this script has been developed for motor-prop simulation

import collections
import time
from turtle import speed
import numpy as np
from collections import deque 


class Motor :
    def __init__(self , init_speed = 0 , max_speed = 700, time_step = 0.001 , zero_to_max_time = 0.02):
        # all radial speeds ore in radian per second 
        self.max_speed = max_speed
        self.motor_speed = deque([init_speed] , maxlen = 10000)
        self.time_step = time_step 
        self.zero_to_max_time = zero_to_max_time
        self.max_change_in_step  = self.max_speed * self.time_step / self.zero_to_max_time
    def reset(self):
        self.motor_speed = deque([0] , maxlen = 10000)

    def action(self , desigred_speed):
        # the action and response of motor based on motor-prop beahvior prediction ()
        speed_diff = desigred_speed - self.motor_speed[-1]
        if speed_diff > 0:
            if speed_diff > self.max_change_in_step :
                self.motor_speed.append(self.motor_speed[-1] + self.max_change_in_step)
            else :
                self.motor_speed.append(desigred_speed)
        elif speed_diff < 0 :
            if -speed_diff > self.max_change_in_step :
                self.motor_speed.append(self.motor_speed[-1] - self.max_change_in_step)
            else :
                self.motor_speed.append(desigred_speed)
    def get_motor_speed(self) :
        curr_speed = self.motor_speed[-1]
        return curr_speed

