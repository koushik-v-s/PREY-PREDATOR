import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs for the Lotka-Volterra model
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z  # z is a vector [prey, predator]
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

# Implement the RK4 method
def rk4_step(f, t, z, h, params):
    k1 = h * f(t, z, *params)
    k2 = h * f(t + h / 2, z + k1 / 2, *params)
    k3 = h * f(t + h / 2, z + k2 / 2, *params)
    k4 = h * f(t + h, z + k3, *params)
    return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Implement the Adaptive Runge-Kutta-Fehlberg (RKF45) method
def rkf45_step(f, t, z, h, tol, params):
    # Butcher tableau coefficients for RKF45
    a = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    b = np.array([[0, 0, 0, 0, 0],
                  [1/4, 0, 0, 0, 0],
                  [3/32, 9/32, 0, 0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0, 0],
                  [439/216, -8, 3680/513, -845/4104, 0],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40]])
    c4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    # Calculate k values
    k = [np.zeros_like(z) for _ in range(6)]
    k[0] = f(t, z, *params)
    for i in range(1, 6):
        k[i] = f(t + a[i] * h, z + h * sum(b[i][j] * k[j] for j in range(i)), *params)

    z4 = z + h * np.dot(c4, k)
    z5 = z + h * np.dot(c5, k)

    # Error estimation
    err = np.linalg.norm(z5 - z4)

    # Adaptive step size control
    if err > tol:
        h *= 0.8 * (tol / err) ** 0.25
        return rkf45_step(f, t, z, h, tol, params)  # Recursively adjust step size
    else:
        h *= 0.9 * (tol / err) ** 0.2
        return z5, h, err

# Solving the system of ODEs
def solve_system(f, z0, t0, tf, h, tol, params, method='rk4'):
    t_values = [t0]
    z_values = [z0]
    errors = []

    t = t0
    z = z0

    while t < tf:
        if method == 'rk4':
            z = rk4_step(f, t, z, h, params)
            t += h
        elif method == 'rkf45':
            z, h, err = rkf45_step(f, t, z, h, tol, params)
            t += h
            errors.append(err)

        t_values.append(t)
        z_values.append(z)

    return np.array(t_values), np.array(z_values), errors

# Function to find the closest index for given time
def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

def run_simulation():
    print("Lotka-Volterra Prey-Predator Model Simulation")

    # Print the differential equations
    print("\nDifferential equations for the Lotka-Volterra model:")
    print("dx/dt = alpha * x - beta * x * y")
    print("dy/dt = delta * x * y - gamma * y")

    # User input for parameters
    alpha = float(input("\nEnter prey growth rate (alpha, e.g., 0.1): "))
    beta = float(input("Enter predation rate (beta, e.g., 0.02): "))
    delta = float(input("Enter predator reproduction rate (delta, e.g., 0.01): "))
    gamma = float(input("Enter predator death rate (gamma, e.g., 0.5): "))

    # Combine parameters into a tuple for passing
    params = (alpha, beta, delta, gamma)

    z0 = np.array([40, 5])  # Initial conditions [prey, predator]
    t0, tf = 0, 200  # Time span
    h = 0.1  # Initial step size
    tol = 1e-6  # Tolerance for RKF45

    # Solve using RK4 method
    t_rk4, z_rk4, _ = solve_system(lotka_volterra, z0, t0, tf, h, tol, params, method='rk4')

    # Solve using RKF45 method
    t_rkf45, z_rkf45, errors_rkf45 = solve_system(lotka_volterra, z0, t0, tf, h, tol, params, method='rkf45')

    # Step 5: Plotting the results
    plt.figure(figsize=(12, 6))

    # RK4 Solution
    plt.subplot(1, 2, 1)
    plt.plot(t_rk4, z_rk4[:, 0], label='Prey (RK4)', color='blue')
    plt.plot(t_rk4, z_rk4[:, 1], label='Predator (RK4)', color='red')
    plt.title("Prey-Predator Dynamics (RK4)")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()

    # RKF45 Solution
    plt.subplot(1, 2, 2)
    plt.plot(t_rkf45, z_rkf45[:, 0], label='Prey (RKF45)', color='blue')
    plt.plot(t_rkf45, z_rkf45[:, 1], label='Predator (RKF45)', color='red')
    plt.title("Prey-Predator Dynamics (RKF45)")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plotting the error for RKF45
    plt.figure(figsize=(6, 4))
    plt.plot(t_rkf45[:-1], errors_rkf45, label='Error (RKF45)', color='green')
    plt.title("Error in RKF45 method")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.yscale('log')
    plt.show()

    # Print the final populations and solution
    print("\nFinal solution using RK4 method:")
    print(f"Prey population: {z_rk4[-1, 0]:.4f}")
    print(f"Predator population: {z_rk4[-1, 1]:.4f}")

    print("\nFinal solution using RKF45 method:")
    print(f"Prey population: {z_rkf45[-1, 0]:.4f}")
    print(f"Predator population: {z_rkf45[-1, 1]:.4f}")

    # Define thresholds for classification
    prey_extinct_threshold = 1
    predator_extinct_threshold = 1
    balanced_threshold = 410  # Example value; adjust based on your model

    # Function to classify the populations
    def classify_population(prey, predator):
        prey_status = "Extinct" if prey < prey_extinct_threshold else "Balanced" if prey < balanced_threshold else "Overpopulated"
        predator_status = "Extinct" if predator < predator_extinct_threshold else "Balanced" if predator < balanced_threshold else "Overpopulated"
        return prey_status, predator_status

    # Add the specified time for t = 200
    specified_times = [0, 2, 4, 6, 8, 10, 200]

    # Print the population as functions of time at specified intervals
    print("\nPopulation as functions of time at specified intervals (0, 2, 4, 6, 8, 10, 200):")

    for t in specified_times:
        idx = find_nearest_index(t_rkf45, t)  # Find the nearest index for the given time
        prey_pop = z_rkf45[idx, 0]
        predator_pop = z_rkf45[idx, 1]
        prey_status, predator_status = classify_population(prey_pop, predator_pop)

        # Calculate dx/dt and dy/dt
        dxdt, dydt = lotka_volterra(t, [prey_pop, predator_pop], *params)

        print(f"Time {t}: Prey = {prey_pop:.4f} ({prey_status}), Predator = {predator_pop:.4f} ({predator_status})")
        print(f"dx/dt = {dxdt:.4f}, dy/dt = {dydt:.4f}")  # Print the derivatives

if __name__ == "__main__":
    run_simulation()


