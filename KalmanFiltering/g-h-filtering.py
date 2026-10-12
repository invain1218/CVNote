"""
g-h filter or alpha-beta filter is a simplified version of the Kalman filter. 
It is used for tracking and smoothing noisy data. 
The g-h filter uses two parameters, g and h, 
to control the influence of the measurement and the prediction on the estimated state.
g is the gain for the measurement update, and h is the gain for the prediction update.
The g-h filter is particularly useful in scenarios where the system dynamics are not well known or are nonlinear, 
and it can be used for real-time applications due to its simplicity and efficiency.


Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state

Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction

Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. New estimate is somewhere on the residual line
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_data(num_points=100, noise_std=1.0):
    """
    Simulate a simple linear motion with Gaussian noise.
    """
    t = np.linspace(0, 10, num_points)
    true_weight = 0.5 * t**2  # True weight follows a quadratic function
    noise = np.random.normal(0, noise_std, size=num_points)
    measured_weight = true_weight + noise
    return t, true_weight, measured_weight

sim = simulate_data(num_points=100, noise_std=5.0)
plt.figure(figsize=(10, 5))
plt.plot(sim[0], sim[1], label='True Weight', color='g')
plt.scatter(sim[0], sim[2], label='Measured Weight', color='r', s=10)
plt.title('Simulated Data with Noise')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.legend()
plt.show()

def g_h_filter(data, x0, dx, g, h, dt):
    """
    Performs g-h filter on 1 state variable with a fixed g and h.
    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step
    """
    x_est = x0
    results = []
    for z in data:
        # Prediction step
        x_pred = x_est + dx * dt # Predict the next state based on the current estimate and change rate

        # Update step
        residual = z - x_pred # correct the prediction with the measurement
        dx = dx + h * residual / dt # Update the change rate based on the residual and h factor
        x_est = x_pred + g * residual # g is the gain for the measurement update

        results.append(x_est)
    return np.array(results)

g_h_results1 = g_h_filter(sim[2], x0=0, dx=0, g=0.8, h=0.2, dt=sim[0][1]-sim[0][0])
g_h_results2 = g_h_filter(sim[2], x0=0, dx=0, g=0.4, h=0.4, dt=sim[0][1]-sim[0][0])

plt.figure(figsize=(10, 5))
plt.plot(sim[0], sim[1], label='True Weight', color='g')
plt.scatter(sim[0], sim[2], label='Measured Weight', color='r', s=10)
plt.plot(sim[0], g_h_results1, label='g-h Filtered Weight (Params 1)', color='b')
plt.plot(sim[0], g_h_results2, label='g-h Filtered Weight (Params 2)', color='orange')
plt.title('g-h Filter Results')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.legend()
plt.show()

