
import numpy as np
import matplotlib.pyplot as plt

# ## === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# and serial timestepping
#
# y = e^{lamda*t}
# dydt = lambda*e^{lamda*t}
# dydt = lambda*y
#
# dy/dt = lambda*y
#
# ## === --- --- === ###

# parameters

nt = 1024
T = 102.4
theta = 0.5

dt = T/nt

q0 = 1

lamda = -0.01 + 1.00j

# setup timeseries

q = np.zeros(nt+1, dtype=complex)

q[0] = q0

def b(t):
    bb = 0
    bb += 2*np.exp(-(t-9.5)*(t-9.5))
    bb += 0.5*np.exp(-(t-21.3)*(t-21.3)/4)
    bb += -5*np.exp(-(t-48.7)*(t-48.7)/9)
    return bb

# timestepping loop

for i in range(nt):
    tn = i*dt
    tn1 = (i+1)*dt

    bb = (1-theta)*b(tn) + theta*b(tn1)

    rhs = (1 + dt*(1-theta)*lamda)*q[i] + dt*bb
    lhs = (1 - dt*theta*lamda)

    q[i+1] = rhs/lhs

# plot
time = np.linspace(0, nt*dt, num=nt+1, endpoint=True)

plt.plot(time, q.real)
plt.show()
