from PID_lib import PID , dif_calculator
import numpy as np
import math
from motor_sim import Motor
from QuadRotor import QuadRotor

import matplotlib.pyplot as plt

quad = QuadRotor(0, 0, 0, 0, 0, 0, 1.2, 1.85, -1.10, 0.2, 0.23, 0.029, 0.022, 0.000004, 300, 300, 300, 300 , False , 0.78)


M1 = Motor()
M2 = Motor()
M3 = Motor()
M4 = Motor()


phi_PID = PID(10000.1965,200.9, 6398.5, 0, 0, 7000, 0.01)
theta_PID = PID(18900.1965, 100.9, 2798.5, 0, 0, 7000, 0.01)
psi_PID = PID(4000.1965, 300.5, 1680.985, 0, 0, 7000, 0.01)

sig = [0]

for i in range(500):
    phi_sig = phi_PID.calculate_command(quad.phi[-1] , 0)
    if phi_sig >700 :
        phi_sig = 700
    if phi_sig < 0 :
        phi_sig = 0

    theta_sig = theta_PID.calculate_command(quad.theta[-1] , 0)
    if theta_sig >700 :
        theta_sig = 700
    if theta_sig < 0 :
        theta_sig = 0

    psi_sig = psi_PID.calculate_command(quad.psii[-1] , 0)
    if psi_sig >700 :
        psi_sig = 700
    if psi_sig < 0 :
        psi_sig = 0


    M1_sig = ((700 - phi_sig)*0.35 +(700 - theta_sig)*0.35  + (psi_sig)*0.3)
    M4_sig = (700 - phi_sig)*0.35 +(theta_sig)*0.35  + (700 - psi_sig)*0.3
    M2_sig = (phi_sig)*0.35+(theta_sig)*0.35  + (psi_sig)*0.3
    M3_sig = (phi_sig)*0.35 +(700 - theta_sig)*0.35  + (700 - psi_sig)*0.3


    M1.action(M1_sig)
    M2.action(M2_sig)
    M3.action(M3_sig)
    M4.action(M4_sig)

    quad.Om1 = M1.get_motor_speed()
    quad.Om2 = M2.get_motor_speed()
    quad.Om3 = M3.get_motor_speed()
    quad.Om4 = M4.get_motor_speed()

    quad.update_factors()
    quad.movement_X_config(np.random.randint(-50,50)/1000,
    np.random.randint(-50,50)/1000,
    np.random.randint(-5,5)/1000)
    sig.append(theta_sig)
    #print(theta_sig)

import numpy as np

plt.plot(quad.theta, 'r-' , label = 'theta')
plt.plot(quad.phi, 'g-' , label = 'phi')
#plt.plot(M2.motor_rpm , label = 'M1')
plt.plot(quad.psii, 'b-' , label = 'psii')

plt.xlabel("time step")
plt.ylabel("radian")
plt.savefig('PID_controller.png')
plt.legend()
plt.show()