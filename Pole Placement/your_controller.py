# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        self.psi_cumulative_error = 0
        self.distance_cumulative_error = 0
        self.psi_previous_error = 0
        self.distance_previous_error = 0

        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        dist, closest_point = closestNode(X, Y, trajectory)

        horizon = 10

        X_des = trajectory[closest_point+horizon, 0]
        Y_des = trajectory[closest_point+horizon, 1]
        psi_des = np.arctan2(Y_des - Y, X_des - X)

        X_C = trajectory[closest_point, 0]
        Y_C = trajectory[closest_point, 1]

        



        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        '''C_alpha = 20000
        m = 1888.6
        lf = 1.55
        lr = 1.39
        Iz = 25854'''

        # Example x_dot value
        #x_dot_value = 10

        # Define the matrix A
        A = np.array([
            [0, 1, 0, 0],
            [0, -4 * Ca / (m * xdot), 4 * Ca / m, -2 * Ca * (lf - lr) / (m * xdot)],
            [0, 0, 0, 1],
            [0, -2 * Ca * (lf - lr) / (Iz * xdot), 2 * Ca * (lf - lr) / Iz, -2 * Ca * (lf**2 + lr**2) / (Iz * xdot)]
        ])

        # Define the matrix B (take the first column to make it a 4x1 matrix)
        B = np.array([
            [0],
            [2 * Ca / m],
            [0],
            [2 * Ca * lf / Iz]
        ])

        # Define desired poles
        desired_poles = np.array([-1, -0.1, -2, -20])

        # Place the poles using scipy.signal.place_poles
        result = signal.place_poles(A, B, desired_poles)

        # Extract the feedback matrix K
        K = result.gain_matrix

        #print("State feedback matrix K:")
        #print(K)
        psi_error = wrapToPi(psi - psi_des)


        numerator = ((Y_des - Y_C) * X - (X_des - X_C) * Y + X_des * Y_C - Y_des * X_C)
        denominator = np.sqrt((Y_des - Y_C)**2 + (X_des - X_C)**2)

        e1 = numerator / denominator
        #e1 = dist
        e2 = psi_error

        e1dot = ydot*np.cos(-e2) - xdot*np.sin(-e2)

        
        
        #e2dot = (psi_error - self.psi_previous_error)/ delT - psidot
        e2dot = psidot
        

        e = np.hstack((e1, e1dot, e2, e2dot)).reshape(4,1)

        delta = wrapToPi(np.dot(-K,e)[0,0])

        self.psi_previous_error = psi_error

        #if (delta < -3.1416 / 6):
         #   delta = -3.1415 / 6
        #elif(delta > 3.1416 / 6):
         #   delta = 3.1416 / 6


        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        distance_error = np.power(np.power(X_des - X, 2)+np.power(Y_des - Y, 2),0.5)
        self.distance_cumulative_error += distance_error*delT

        Kdp = 500
        Kdd = 0.01
        Kdi = 0.3

        F = Kdp*distance_error + Kdd*(distance_error - self.distance_previous_error)/delT + Kdi*self.distance_cumulative_error

        self.distance_previous_error = distance_error

        if (F < 0):
            F = 0
        elif(F > 15736):
            F = 15736

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
