#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:19:33 2022

@author: hesam
"""

from PID_lib import PID 
import numpy as np
import math
from motor3 import Motor
from Quadrotor import QuadRotor
import matplotlib.pyplot as plt 
import math
import collections
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
class action_spacer():
    def __init__(self , action_space_lower , action_space_higher):
        self.high = action_space_higher

def action_aggregator(a1, a2 , a3):
    # in "X" con]fig Quadrotor, a1 is controlling roll(phi) angel 
    # required action shape = [p1 , p2 , p3 , p4]
    #a1 , a2 , a3 are float numbers in [-1,1] 
    # normalize actions between [0,700]
    c1 = 0.4
    c2 = 0.4
    c3 = 0.2
    a1 = (a1+1)/2
    a2 = (a2+1)/2
    a3 = (a3+1)/2
    M1_sig = (1 - a1)*c1 +(1 - a2)*c2  + (a3)*c3
    M4_sig = (1 - a1)*c1 +(a2)*c2      + (1 - a3)*c3
    M2_sig = (a1)*c1     +(a2)*c2      + (a3)*c3
    M3_sig = (a1)*c1     +(1 - a2)*c2  + (1 - a3)*c3
    action = [M1_sig , M2_sig , M3_sig  , M4_sig]
    return action
class QuadRotorEnv :
    def __init__(self, state_dim, state_hist, action_dim,
                 max_episode_steps, max_action, min_action,
                 max_acceptable_angel,rnn_mode = False, multi_agent = False):
        self.state_dim = state_dim  
        self.state_dim = [4+state_hist*6]
        self.action_dim = action_dim
        self.curr_step = 0
        self.rnn_mode = rnn_mode
        if multi_agent :
            print("zjg")
            self.state_dim = [4 + state_hist*2]
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.max_action = max_action
        self.min_action= min_action
        self.state_hist = state_hist # use this much of history of angels as input of agent
        self.max_acceptable_angel = max_acceptable_angel
        
        self.M1 = Motor()
        self.M2 = Motor()
        self.M3 = Motor()
        self.M4 = Motor()
        #init state
        init_phi = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.5)
        init_theta = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.5)
        init_psi = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.5)
        
        init_p = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.0)
        init_q = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.0)
        init_r = np.append(np.zeros(int(state_hist-1), dtype = np.float32), 0.0)
        
        self.quad = QuadRotor(init_p, init_q, init_r, 0, 0, 0, init_phi, init_theta, init_psi,
                                  0.4, 0.04, 0.05, 0.022, 0.000004, 0, 0, 0, 0 , False , 0.78)

        self.reward_history = []
        self.state_history = []
        
        self.action_space = action_spacer(0,1)

    
    def step_singel_agent(self, action):
        #acts an action on singel agent controller mode
        #action is a [1*4] list, eachaction is in range(-1,1)
        self.M1.action((action[0]+1)*350)
        self.M2.action((action[1]+1)*350)
        self.M3.action((action[2]+1)*350)
        self.M4.action((action[3]+1)*350)
        self.quad.Om1 = self.M1.get_motor_speed()
        self.quad.Om2 = self.M2.get_motor_speed()
        self.quad.Om3 = self.M3.get_motor_speed()
        self.quad.Om4 = self.M4.get_motor_speed()
        self.quad.update_factors()
        self.quad.movement_X_config(0,0,0)
        
        new_state = [self.quad.phi[-self.state_hist:] , self.quad.theta[-self.state_hist:] , self.quad.psii[-self.state_hist:],
                     self.quad.p[-self.state_hist:], self.quad.q[-self.state_hist:] , self.quad.r[-self.state_hist:],
                     [self.M1.motor_speed[-1]/700] , [self.M2.motor_speed[-1]/700] , [self.M3.motor_speed[-1]/700] , [self.M4.motor_speed[-1]/700]]
        if self.rnn_mode :
            new_state = [flatten(new_state[0]), flatten(new_state[1]), 
            flatten(new_state[2]), flatten(new_state[3]), 
            flatten(new_state[4]), flatten(new_state[5])]
        else :
            new_state = flatten(new_state)
        

        reward = 0
        is_done = False
        #reward = (1-(abs(self.quad.phi[-1])) + (1-abs(self.quad.theta[-1]) )+ (1-abs(self.quad.psii[-1])))
        """reward = (-(abs(self.quad.phi[-1])*10)**2 -
                  (abs(self.quad.theta[-1])*10)**2 -
                  (abs(self.quad.psii[-1])*10)**2)"""
        reward = ((1.5-(abs(self.quad.phi[-1])))**5 +
                  (1.5-(abs(self.quad.theta[-1])))**5+
                  (1.5-(abs(self.quad.psii[-1])))**5) 
        if (self.quad.phi[-1] > self.max_acceptable_angel or
            self.quad.phi[-1] <-self.max_acceptable_angel or
            self.quad.theta[-1] > self.max_acceptable_angel or
            self.quad.theta[-1] <-self.max_acceptable_angel or
            self.quad.psii[-1] > self.max_acceptable_angel or
            self.quad.psii[-1] <-self.max_acceptable_angel  ):
            is_done = True 
            reward -= 10
        
        self.curr_step += 1
        if (self.curr_step >= self.max_episode_steps) :
            print("max steps reached " , self.curr_step )
            is_done = True
        
        return new_state , reward , is_done , []
    def step_Multi_agent(self, action):
        #acts an action on singel agent controller mode
        #action is a [1*4] list, eachaction is in range(-1,1)
        self.M1.action((action[0])*700)
        self.M2.action((action[1])*700)
        self.M3.action((action[2])*700)
        self.M4.action((action[3])*700)

        print("motor1 actin is : ", (action[0]+1)*350)
        self.quad.Om1 = self.M1.get_motor_speed()
        self.quad.Om2 = self.M2.get_motor_speed()
        self.quad.Om3 = self.M3.get_motor_speed()
        self.quad.Om4 = self.M4.get_motor_speed()
        self.quad.update_factors()
        self.quad.movement_X_config(0,0,0)
        
        new_state = [self.quad.phi[-self.state_hist:] , self.quad.theta[-self.state_hist:] , self.quad.psii[-self.state_hist:],
                     self.quad.p[-self.state_hist:], self.quad.q[-self.state_hist:] , self.quad.r[-self.state_hist:],
                     [self.M1.motor_speed[-1]/700] , [self.M2.motor_speed[-1]/700] , [self.M3.motor_speed[-1]/700] , [self.M4.motor_speed[-1]/700]]
        new_state = [flatten([new_state[0],new_state[3],new_state[-4:]]),
                     flatten([new_state[1],new_state[4],new_state[-4:]]),
                     flatten([new_state[2],new_state[5],new_state[-4:]])]
        #new_state = flatten(new_state)
        # new_state = np.array(new_state)
        # new_state = new_state.flatten()
        
        reward = [0,0,0]
        is_done = False
        #reward = (1-(abs(self.quad.phi[-1])) + (1-abs(self.quad.theta[-1]) )+ (1-abs(self.quad.psii[-1])))
        """reward = (-(abs(self.quad.phi[-1])*10)**2 -
                  (abs(self.quad.theta[-1])*10)**2 -
                  (abs(self.quad.psii[-1])*10)**2)"""
        reward_phi = (1.5-(abs(self.quad.phi[-1])))**5
        reward_theta = (1.5-(abs(self.quad.theta[-1])))**5
        reward_psii = (1.5-(abs(self.quad.psii[-1])))**5
        reward[0] = reward_phi
        reward[1] = reward_theta
        reward[2] = reward_psii
        if (self.quad.phi[-1] > self.max_acceptable_angel or
            self.quad.phi[-1] <-self.max_acceptable_angel):
            is_done = True 
        if (self.quad.theta[-1] > self.max_acceptable_angel or
            self.quad.theta[-1] <-self.max_acceptable_angel):
            is_done = True 
        if (self.quad.psii[-1] > self.max_acceptable_angel or
            self.quad.psii[-1] <-self.max_acceptable_angel):
            is_done = True 

        if (self.curr_step >= self.max_episode_steps) :
            print("max steps reached " , self.curr_step )
            is_done = True
        
        self.curr_step += 1

        print(self.curr_step)
        return new_state , reward , is_done , []
    def get_state(self):
        state = [self.quad.phi[-1] , self.quad.theta[-1] , self.quad.psii[-1], self.quad.p[-1] , self.quad.q[-1] , self.quad.r[-1] , self.M1.motor_speed[-1]/700 , self.M2.motor_speed[-1]/700 , self.M3.motor_speed[-1]/700 , self.M4.motor_speed[-1]/700]
        return state

    def reset(self):
        init_phi = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        init_theta = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        init_psi = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        
        init_p = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        init_q = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        init_r = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        
        self.quad = QuadRotor(init_p[-1], init_q[-1], init_r[-1] , 0, 0, 0, init_phi[-1], init_theta[-1], init_psi[-1],
                                  0.2, 0.02, 0.019, 0.022, 0.000004, 0, 0, 0, 0 , False , 0.78)
        self.quad = QuadRotor(init_p, init_q, init_r, 0, 0, 0, init_phi, init_theta, init_psi,
                                  0.2, 0.02, 0.019, 0.022, 0.000004, 0, 0, 0, 0 , False , 0.78)
        self.M1.reset()
        self.M2.reset()
        self.M3.reset()
        self.M4.reset()
        self.curr_step = 0
        re = [init_phi, init_theta, init_psi, init_p, init_q, init_r,0,0,0,0]
        re = flatten(re)
        return re
    def reset_multi_agent(self):
        init_phi = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        init_theta = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        init_psi = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), np.random.randint(-100,100)/200)
        
        init_p = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        init_q = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        init_r = np.append(np.zeros(int(self.state_hist-1), dtype = np.float32), 0.0)
        
        self.quad = QuadRotor(init_p[-1], init_q[-1], init_r[-1] , 0, 0, 0, init_phi[-1], init_theta[-1], init_psi[-1],
                                  0.4, 0.04, 0.05, 0.022, 0.000004, 0, 0, 0, 0 , False , 0.78)
        self.quad = QuadRotor(init_p, init_q, init_r, 0, 0, 0, init_phi, init_theta, init_psi,
                                  0.4, 0.04, 0.05, 0.022, 0.000004, 0, 0, 0, 0 , False , 0.78)
        self.M1.reset()
        self.M2.reset()
        self.M3.reset()
        self.M4.reset()
        self.curr_step = 0
        re = [[init_phi,init_p,0,0,0,0], [init_theta,init_q,0,0,0,0], [init_psi, init_r,0,0,0,0],0,0,0,0]
        #re = flatten(re)
        return re
    def get_sample_action(self):
        re = [np.random.randint(-3500,3500)/10, np.random.randint(-3500,3500)/10 , np.random.randint(-3500,3500)/10 , np.random.randint(-3500,3500)/10]
        #print("random action is : " , re)
        return re