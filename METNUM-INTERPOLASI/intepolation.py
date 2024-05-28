import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Interpolasi Lagrange
def lagrange_interpolation(x, y, x_new):
    def L(k, x_new):
        L_k = [(x_new - x[j]) / (x[k] - x[j]) for j in range(len(x)) if j != k]
        return np.prod(L_k)

    return sum(y[k] * L(k, x_new) for k in range(len(x)))

# Interpolasi Newton
def newton_interpolation(x, y, x_new):
    def divided_diff(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y

        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

        return coef[0, :] 

    coef = divided_diff(x, y)
    n = len(coef)
    y_new = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x_new - x[j])
        y_new += term
    return y_new

# Generate data for plotting
x_vals = np.linspace(5, 40, 100)
y_lagrange = [lagrange_interpolation(x, y, xi) for xi in x_vals]
y_newton = [newton_interpolation(x, y, xi) for xi in x_vals]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_vals, y_lagrange, label='Lagrange interpolation')
plt.plot(x_vals, y_newton, label='Newton interpolation')
plt.xlabel('Tegangan (kg/mm²)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.title('Interpolasi Lagrange dan Newton')
plt.grid(True)
plt.show()
