# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:37:59 2022

@author: hesam ghashami
"""
import math
import matplotlib.pyplot as plt
import cv2 
import os

def dif_calculator(X , Y , a):
	"""
	X : left motor speed (increased by a )
	Y : right motor speed ( will be decreased by result of this function)
	"""
	if a > 0 :
		c = Y - math.sqrt(Y**2 - a**2 - 2*X*a)
		return c
	else :
		c = -Y + math.sqrt(Y**2 - a**2 - 2*X*a)
		return c
	
	
	
class PID:
	def __init__(self , K_p , K_i , K_d ,init_angle, min_command , max_command  , time_step):
		self.K_d = K_d 
		self.K_p = K_p 
		self.K_i = K_i
		
		self.P_error = [0]
		self.I_error = [0]
		self.D_error = [0]
		
		self.min_command = min_command
		self.max_command = max_command
		
		self.angle_history = [init_angle]
		self.time_step = time_step
		
	def calculate_command(self , current , desired):
		self.angle_history.append(current)
		p_error = desired - current
		self.P_error.append(p_error)
		if len(self.P_error)>4:
			d_error = (self.P_error[-1] - self.P_error[-2])/(self.time_step)
			self.D_error.append(d_error)
		else :
			d_error = 0
		
		I_error = self.I_error[-1] + p_error*self.time_step
		self.I_error.append(I_error)
		
		output_signal = p_error*self.K_p + d_error*self.K_d + I_error*self.K_i 
		return output_signal
	
		
		