import numpy as np
import scipy.signal as signal

# Define constants
C_alpha = 20000
m = 1888.6
lf = 1.55
lr = 1.39
Iz = 25854

# Example x_dot value
x_dot_value = 10

# Define the matrix A
A = np.array([
    [0, 1, 0, 0],
    [0, -4 * C_alpha / (m * x_dot_value), 4 * C_alpha / m, -2 * C_alpha * (lf - lr) / (m * x_dot_value)],
    [0, 0, 0, 1],
    [0, -2 * C_alpha * (lf - lr) / (Iz * x_dot_value), 2 * C_alpha * (lf - lr) / Iz, -2 * C_alpha * (lf**2 + lr**2) / (Iz * x_dot_value)]
])

# Define the matrix B (take the first column to make it a 4x1 matrix)
B = np.array([
    [0],
    [2 * C_alpha / m],
    [0],
    [2 * C_alpha * lf / Iz]
])

# Define desired poles
desired_poles = np.array([-2, -3, -4, -5])

# Place the poles using scipy.signal.place_poles
result = signal.place_poles(A, B, desired_poles)

# Extract the feedback matrix K
K = result.gain_matrix

print("State feedback matrix K:")
print(K)
