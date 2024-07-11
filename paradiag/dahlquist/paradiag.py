
import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

# ## === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# solving the all-at-once system with ParaDiag
#
# y = e^{lamda*t}
# dydt = lambda*e^{i*lamda*t}
# dydt = lambda*y
#
# dy/dt = lambda*y
#
# ## === --- --- === ###

# parameters

verbose = False

alpha = 1e-0

T = 102.4
nt = 1024
theta = 0.5

dt = T/nt

y0 = 1

lamda = -0.01 + 1.00j

dtype = complex

time = np.linspace(dt, nt*dt, num=nt, endpoint=True)

# ## timestepping toeplitz matrices

# # mass toeplitz

# first column
b1 = np.zeros(nt, dtype=dtype)
b1[0] = 1/dt
b1[1] = -1/dt

# first row
r1 = np.zeros_like(b1)
r1[0] = b1[0]

# # function toeplitz

# first column
b2 = np.zeros(nt, dtype=dtype)
b2[0] = -lamda*theta
b2[1] = -lamda*(1-theta)

# first row
r2 = np.zeros_like(b2)
r2[0] = b2[0]

# ## all-at-once system

# # Jacobian

acol = b1 + b2
arow = r1 + r2


class ToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, col, row):
        self.dtype = col.dtype
        self.shape = tuple((len(col), len(row)))

        self.col = col
        self.row = row
        self.mat = tuple((col, row))

    def _matvec(self, v):
        return linalg.matmul_toeplitz(self.mat, v)


A = ToeplitzLinearOperator(acol, arow)

# ## paradiag preconditioner


class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col):
        self.dtype = col.dtype
        self.col = col
        n = len(col)
        self.shape = tuple((n, n))
        self.eigvals = fft(col, norm='backward')

    def _matvec(self, v):
        return ifft(fft(v)/self.eigvals)


class AlphaCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, alpha=1):
        self.dtype = col.dtype
        self.alpha = alpha
        self.col = col
        n = len(col)
        self.shape = tuple((n, n))

        # fft weighting
        self.gamma = alpha**(np.arange(n)/n)

        # eigenvalues
        self.eigvals = fft(col*self.gamma, norm='backward')

    def _to_eigvecs(self, v):
        return fft(v*self.gamma, norm='ortho')

    def _from_eigvecs(self, v):
        return ifft(v, norm='ortho')/self.gamma

    def _matvec(self, v):
        return self._from_eigvecs(self._to_eigvecs(v)/self.eigvals)


# P = CirculantLinearOperator(acol)
P = AlphaCirculantLinearOperator(acol, alpha=alpha)

# forcing

def b(t):
    bb = 0
    bb += 2*np.exp(-(t-9.5)*(t-9.5))
    bb += 0.5*np.exp(-(t-21.3)*(t-21.3)/4)
    bb += -5*np.exp(-(t-48.7)*(t-48.7)/9)
    return bb

# ## right hand side

rhs = np.zeros(nt, dtype=dtype)

rhs[0] += -(b1[1] + b2[1])*y0 + (1-theta)*b(0)

rhs[1:] += (1-theta)*b(time[:-1])
rhs[:] += theta*b(time[:])


# ## residual

def residual(x):
    return rhs - A.matvec(x)


# ## solve all-at-once system

niterations = 0


def gmres_callback(pr_norm):
    global niterations
    print(f"niterations: {str(niterations).rjust(5,' ')} | residual: {pr_norm}")
    niterations += 1
    return


y, exit_code = spla.gmres(A, rhs, M=P,
                          #restart=nt,
                          #tol=1e-14, atol=1e-14,
                          callback=gmres_callback,
                          callback_type='pr_norm')

print(f"gmres exit code: {exit_code}")
print(f"gmres iterations: {niterations}")
print(f"residual: {linalg.norm(residual(y))}")

plt.plot(time, y.real)
plt.show()
