# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:46:27 2022

@author: ASUS
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 19:01:18 2022

@author: ASUS
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:59:02 2022

@author: Hesam ghashami
"""


from math import sin , cos , tan , pi 
import numpy as np
import os
import matplotlib.pyplot as plt

# quadrotor simulation 
class QuadRotor:
    def __init__(self  , p ,q, r ,p_d  ,q_d,r_d, phi , theta , psii , l , I_xx , I_yy , I_zz , J_r , Om1 , Om2 , Om3 , Om4 , apply_constraitions , max_ang ) :
        self.p = [p] 
        self.q = [q] 
        self.r = [r] 
        self.p_d = [p_d] 
        self.q_d = [q_d] 
        self.r_d = [r_d] 
        self.phi = [phi] 
        self.theta = [theta] 
        self.psii = [psii]
        self.phi_d = [0] 
        self.theta_d = [0] 
        self.psii_d = [0] 
        self.l = l 
        self.l_p = l *0.707106781
        self.I_xx = I_xx 
        self.I_yy = I_yy 
        self.I_zz = I_zz 
        self.J_r = J_r 
        self.Om1 = Om1 
        self.Om2 = Om2 
        self.Om3 = Om3 
        self.Om4 = Om4 
        self.U2= 0
        self.U3 = 0  
        self.U4 = 0 
        self.Omega =  0
        
        self.K1 = 0.00002 
        self.K2 = 0.00001 
        self.time_step = 0.01 
        
		
        # thruster and gyro tourques in body frame
        self.T_m_x = 0
        self.T_m_y = 0 
        self.T_m_z = 0
		
        self.apply_constraitions = apply_constraitions
        self.max_ang = max_ang
    def movement_X_config(self , ext_x , ext_y , ext_z):
        p_d = (self.U_phi/self.I_xx)+((self.q[-1] * self.r[-1])*(self.I_yy - self.I_zz)/self.I_xx) - ((self.J_r * self.q[-1]*self.Omega)/self.I_xx)  + (ext_x/self.I_xx)
        q_d = (self.U_theta/self.I_yy)+((self.p[-1] * self.r[-1])*(self.I_zz - self.I_xx)/self.I_yy) - ((self.J_r * self.p[-1]*self.Omega)/self.I_yy) + (ext_y/self.I_yy)
        r_d = (self.U_psi/self.I_zz) + ((self.p[-1] * self.q[-1])*(self.I_xx - self.I_yy)/self.I_zz) + (ext_z/self.I_zz)

        ################################################
        # calculation of angular velocity by integration of angular acceleration 
        p = self.p[-1] + p_d*self.time_step 
        q = self.q[-1] + q_d*self.time_step 
        r = self.r[-1] + r_d*self.time_step 
        
        self.p.append(p)
        self.q.append(q)
        self.r.append(r)
        ###############################################
        """
		# phi , theta and psii are not euler angels ###
		"""
        phi = (0.5* p_d * self.time_step**2) + p*self.time_step + self.phi[-1]
        self.phi.append(phi)
		
        theta = (0.5* q_d * self.time_step**2) + q*self.time_step  + self.theta[-1]
        self.theta.append(theta)
		
        psii = (0.5* r_d * self.time_step**2) + r*self.time_step  + self.psii[-1]
        self.psii.append(psii)
		

	
        if self.apply_constraitions == True :
            if self.phi[-1] > self.max_ang :
                self.phi[-1] = self.max_ang
                self.p_d[-1] = 0
                self.p[-1] = 0
                
            if self.phi[-1] < -self.max_ang :
                self.phi[-1] = -self.max_ang
                self.p_d[-1] = 0
                self.p[-1] = 0
				
				
            if self.theta[-1] > self.max_ang :
                self.phi[-1] = self.max_ang
                self.theta[-1] = 0
                self.q[-1] = 0
                
            if self.theta[-1] < -self.max_ang :
                self.theta[-1] = -self.max_ang
                self.theta[-1] = 0  
                self.q[-1] = 0  

    def movement(self , ext_x , ext_y , ext_z ):
        # # # # #
        ## calculate angular acceleration
        p_d = (self.q[-1]*self.r[-1]*(self.I_yy - self.I_zz)/self.I_xx)  - (self.J_r * self.q[-1]*self.Omega/self.I_xx) + (self.l*self.U2/self.I_xx) + (ext_x/self.I_xx)
        q_d = (self.p[-1]*self.r[-1]*(self.I_zz - self.I_xx)/self.I_yy) + (self.J_r * self.p[-1]*self.Omega/self.I_yy) + (self.l*self.U3/self.I_yy) + (ext_y/self.I_yy)
        r_d =  (self.p[-1]*self.q[-1]*(self.I_xx - self.I_yy)/self.I_zz) + self.U4/self.I_zz + (ext_z/self.I_zz)


        ################################################
        # calculation of angular velocity by integration of angular acceleration 
        p = self.p[-1] + p_d*self.time_step 
        q = self.q[-1] + q_d*self.time_step 
        r = self.r[-1] + r_d*self.time_step 
        
        self.p.append(p)
        self.q.append(q)
        self.r.append(r)
        ###############################################
        """
		# phi , theta and psii are not euler angels ###
		"""
        phi = (0.5* p_d * self.time_step**2) + p*self.time_step + self.phi[-1]
        self.phi.append(phi)
		
        theta = (0.5* q_d * self.time_step**2) + q*self.time_step  + self.theta[-1]
        self.theta.append(theta)
		
        psii = (0.5* r_d * self.time_step**2) + r*self.time_step  + self.psii[-1]
        self.psii.append(psii)
		

	
        if self.apply_constraitions == True :
            if self.phi[-1] > self.max_ang :
                self.phi[-1] = self.max_ang
                self.p_d[-1] = 0
                self.p[-1] = 0
                
            if self.phi[-1] < -self.max_ang :
                self.phi[-1] = -self.max_ang
                self.p_d[-1] = 0
                self.p[-1] = 0
				
				
            if self.theta[-1] > self.max_ang :
                self.phi[-1] = self.max_ang
                self.theta[-1] = 0
                self.q[-1] = 0
                
            if self.theta[-1] < -self.max_ang :
                self.theta[-1] = -self.max_ang
                self.theta[-1] = 0  
                self.q[-1] = 0           
        
        # for adding limits for euler angels an adjustment for assignmets of variables should be considered
    def update_factors(self) :
        self.U2 = self.K1*(self.Om4**2 - self.Om2**2)
        self.U3 = self.K1*(self.Om3**2 - self.Om1**2) 
        self.U4 = self.K2*(self.Om4**2 + self.Om2**2 - self.Om3**2 - self.Om1**2 )
        self.U_phi = self.K1*(self.Om2**2+ self.Om3**2- self.Om1**2 -self.Om4**2  )* self.l_p
        self.U_theta =  self.K1*(self.Om2**2- self.Om3**2 - self.Om1**2 + self.Om4**2 )* self.l_p
        self.U_psi = self.K2*(self.Om1**2 + self.Om2**2 - self.Om3**2 - self.Om4**2 )
        self.Omega = self.Om4 + self.Om2 + self.Om1 + self.Om3 


# myQuad = QuadRotor(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.02, 0.019, 0.022, 0.000004, 700, 700.0, 700, 700 , False , 0.78)

# for i in range(10):
#  	myQuad.update_factors()
#  	myQuad.movement(0.1, 0, 0)


# plt.plot(myQuad.p)
# plt.plot(myQuad.q)
# plt.plot(myQuad.r)
# plt.show()
# plt.plot(myQuad.phi)
# plt.plot(myQuad.theta)
# plt.plot(myQuad.psii)
	