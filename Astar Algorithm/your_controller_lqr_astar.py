# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
from scipy.signal import StateSpace, lsim, dlsim
import scipy.linalg

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
        #self.distance_cumulative_error = 0
        self.psi_previous_error = 0
        #self.distance_previous_error = 0
        #self.integralPsiError = 0
        #self.previousPsiError = 0
        self.previousXdotError = 0

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        # ---------------|Lateral Controller|-------------------------
        _, node = closestNode(X, Y, trajectory)
        # Choose a node that is ahead of our current node based on index
        forwardIndex = 100
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

        C = np.eye(4)
        D = np.array([
            [0],
            [0],
            [0],
            [0]
        ])

        sys_ct = StateSpace(A, B, C, D)
        sys_dt = sys_ct.to_discrete(delT)

        A_d = sys_dt.A
        B_d = sys_dt.B
        
        Q = np.array([
            [20, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 20]
        ])

        R = np.array([
            [33.5],
        ])

        S = np.matrix(scipy.linalg.solve_discrete_are(A_d, B_d, Q, R))
        #compute the LQR gain
        K = -np.matrix(scipy.linalg.inv(B_d.T@S@B_d+R)@(B_d.T@S@A_d))

        #psi_error = wrapToPi(psi - psi_des)

        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1],trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 = (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)
        except:
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], trajectory[-1,0]-trajectory[node,0])
            e1 = (Y - trajectory[-1,1])*np.cos(psiDesired) - (X - trajectory[-1,0])*np.sin(psiDesired)
        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot # This definition would be psidot - psidotDesired if calculated from curvature
            # Assemble error-based states into array
        states = np.array([e1,e1dot,e2,e2dot])
            # Calculate delta via u = -Kx
        delta = float(K @ states)


        
        '''e1 = np.power(np.power(X_des - X, 2)+np.power(Y_des - Y, 2),0.5)
        
        e2 = psi_error

        
        e1dot = ydot + xdot * (e2)

        
        
        
        e2dot = psidot
        

        e = np.hstack((e1, e1dot, e2, e2dot)).reshape(4,1)

        delta = wrapToPi(np.dot(-K,e)[0,0])

        if (delta < -np.pi / 6):
            delta = -np.pi / 6
        elif(delta > np.pi / 6):
            delta = np.pi / 6'''

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
        kp = 200
        ki = 10
        kd = 30
        desiredVelocity = 8
        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        if (F < 0):
            F = 0
        elif(F > 15736):
            F = 15736


        # Return all states and calculated control inputs (F, delta) and obstacle position
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
