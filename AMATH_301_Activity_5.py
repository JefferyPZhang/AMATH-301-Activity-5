import numpy as np
import matplotlib.pyplot as plt
import scipy

data = np.genfromtxt('data.csv', delimiter = ',')
x = data[0, :]
y = data[1, :]


def f(x, coeffs = [.98, 1.02, 1.5]):
    return coeffs[0] * x * np.cos(coeffs[1] * x) ** 2 + coeffs[2] + np.random.normal(0, 0.01, x.shape)

def f_coeffs(x, coeffs):
    return coeffs[0] * x * np.cos(coeffs[1] * x) ** 2 + coeffs[2]

def f_prime(x, coeffs = [.98, 1.02, 1.5]):
    return coeffs[0] * np.cos(coeffs[1] * x) * (np.cos(coeffs[1] * x) - 2 * coeffs[1] * x * np.sin(coeffs[1] * x))

def f_int(x, coeffs = [.98, 1.02, 1.5]):
    return coeffs[2] * x + (coeffs[0] * x ** 2) / 4 + (coeffs[0] * np.cos(2 * coeffs[1] * x)) / (8 * coeffs[1] ** 2) + (coeffs[0] * x * np.sin(2 * coeffs[1] * x)) / (4 * coeffs[1])

# Approximation Methods

true_func = np.zeros(len(x))
for i in range (len(x) - 1):
    true_func[i] = f_prime(x[i])

def forward_sec(i):
    return (y[i + 1] - y[i]) / A1

def backward_sec(i):
    return (y[i] - y[i - 1]) / A1

def center_sec(i):
    return (y[i + 1] - y[i - 1]) / (A1 * 2)

A1 = x[1] - x[0]
A2 = np.zeros(len(x))

for i in range (len(A2) - 1):
    A2[i] = forward_sec(i)
    
A3 = np.linalg.norm(A2 - true_func)
A4 = np.zeros(len(x))

for i in range (len(A4) - 1):
    A4[i] = backward_sec(i) 
    
A5 = np.linalg.norm(A4 - true_func)
A6 = np.zeros(len(x))

for i in range (len(A6) - 1):
    A6[i] = center_sec(i)
    
A2[len(A2) - 1] = (y[len(x) - 1] - y[len(x) - 2]) / A1
A4[0] = A2[0]
A4[len(A2) - 1] = (y[len(x) - 1] - y[len(x) - 2]) / A1
A6[len(A2) - 1] = (y[len(x) - 1] - y[len(x) - 2]) / A1
A6[0] = A2[0]
A7 = np.linalg.norm(A6 - f_prime(x))

print(A3)
print(A5)
print(A7)

coeffs_poly = np.polyfit(x, y, 8)
f_prime_polyfit = np.polyder(coeffs_poly)
yplot_polyder = np.polyval(f_prime_polyfit, x)

A8 = np.linalg.norm(yplot_polyder - f_prime(x))

coeffs = ([1, 1, 1])
Sum_Squared_Error = lambda coeffs: np.sum((np.abs(coeffs[0] * x * np.cos(coeffs[1] * x) ** 2 + coeffs[2] - y)) ** 2)
coeffs_min = scipy.optimize.fmin(Sum_Squared_Error, coeffs)

A9 = Sum_Squared_Error([1, 1, 1])
A10 = coeffs_min
A11 = np.linalg.norm(f_coeffs(x, coeffs_min) - f(x))
'''
plt.plot(x, y, label = "data")
plt.plot(x, A2, label = "forward")
plt.plot(x, A4, label = "backward")
plt.plot(x, A6, label = "centered")
plt.plot(x, f_prime(x), label = "true deriv")
plt.legend()
plt.show()
'''
def LHR(x, y, delta_x):
    LHR = 0
    for k in range(len(x) - 1):
        LHR = LHR + delta_x * y[k]
    return LHR
def RHR(x, y, delta_x):
    RHR = 0
    for k in range(1, x.size):
        RHR = RHR + delta_x * y[k]
    return RHR
def Trap(x, y, delta_x):
    return (LHR(x, y, delta_x) + RHR(x, y, delta_x)) / 2

A12 = LHR(x, y, A1)
A13 = RHR(x, y, A1)
A14 = Trap(x, y, A1)

coeffs_poly = np.polyfit(x, y, 8)
f_int_polyfit = np.polyint(coeffs_poly)
print(f_int_polyfit)

A16 = f_int_polyfit
A17 = f_int_polyfit
A18 = f_int_polyfit