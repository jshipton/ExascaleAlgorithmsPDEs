import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from math import pi
import argparse

# ## === --- --- === ###
#
# Solve the linear advection diffusion equation
#   using sequential timestepping
#
# ADE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: centred finite differences
#
# ## === --- --- === ###


# Finite difference spatial discretisations
def gradient_stencil(grad, order):
    '''
    Return the centred stencil for the `grad`-th gradient
    of order of accuracy `order`
    '''
    return {
        1: {  # first gradient
            2: np.array([-1/2, 0, 1/2]),
            4: np.array([1/12, -2/3, 0, 2/3, -1/12]),
            6: np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
        },
        2: {  # second gradient
            2: np.array([1, -2, 1]),
            4: np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
            6: np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        },
        4: {  # fourth gradient
            2: np.array([1,  -4, 6, -4, 1]),
            4: np.array([-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]),
            6: np.array([7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240])  # noqa: E501
        }
    }[grad][order]


def sparse_circulant(stencil, n):
    '''
    Return sparse scipy matrix from finite difference
    stencil on a periodic grid of size n.
    '''
    if len(stencil) == 1:
        return sparse.spdiags([stencil[0]*np.ones(n)], 0)

    # extend stencil to include periodic overlaps
    ns = len(stencil)
    noff = (ns-1)//2
    pstencil = np.zeros(ns+2*noff)

    pstencil[noff:-noff] = stencil
    pstencil[:noff] = stencil[noff+1:]
    pstencil[-noff:] = stencil[:noff]

    # constant diagonals of stencil entries
    pdiags = np.tile(pstencil[:, np.newaxis], n)

    # offsets for inner domain and periodic overlaps
    offsets = np.zeros_like(pstencil, dtype=int)

    offsets[:noff] = [-n+1+i for i in range(noff)]
    offsets[noff:-noff] = [-noff+i for i in range(2*noff+1)]
    offsets[-noff:] = [n-noff+i for i in range(noff)]

    return sparse.spdiags(pdiags, offsets)


parser = argparse.ArgumentParser(
    description="Serial-in-time advection-diffusion equation with implicit theta method",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nt', type=int, default=128, help="Number of timesteps.")
parser.add_argument('--nx', type=int, default=128, help="Number of spatial nodes.")
parser.add_argument('--lx', type=float, default=2*pi, help="Length of domain.")
parser.add_argument('--u', type=float, default=1, help="Advecting velocity.")
parser.add_argument('--re', type=float, default=500, help="Reynolds number.")
parser.add_argument('--cfl', type=float, default=0.8, help="Courant number.")
parser.add_argument('--theta', type=float, default=0.5, help="Implicit parameter.")
parser.add_argument('--plot', action='store_true', help="Plot timeseries.")
parser.add_argument('--plot_freq', type=int, default=16, help="Frequency of timesteps to plot.")
parser.add_argument('--show_args', action='store_true', help="Print value of arguments.")

args = parser.parse_args()

if args.show_args:
    print(args)

np.set_printoptions(linewidth=100, precision=3)

# number of time and space points
nt, nx, lx = args.nt, args.nx, args.lx
dx = lx/nx

mesh = np.linspace(start=-lx/2, stop=lx/2, num=nx, endpoint=False)

# velocity and reynolds number
u, re, cfl, theta = args.u, args.re, args.cfl, args.theta

# timestep
nu = lx*u/re
dt = cfl*dx/u

# Courant numbers
cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx
print(f"{cfl_u = }")
print(f"{cfl_v = }")

# Mass matrix
M = sparse_circulant([1], nx)

# Advection matrix
D = sparse_circulant(gradient_stencil(1, order=2), nx)

# Diffusion matrix
L = sparse_circulant(gradient_stencil(2, order=2), nx)

# Spatial terms
K = (u/dx)*D - (nu/dx**2)*L

# A0*q^{n} + A1*q^{n+1} = 0
A0 = -M/dt + (1 - theta)*K
A1 = M/dt + theta*K
A1.solve = spla.factorized(A1.tocsc())

# initial conditions
qinit = np.zeros_like(mesh)
qinit[:] = np.cos(mesh/2)**4

# calculate timeseries
q = np.zeros((nt+1, len(qinit)))
q[0] = qinit

for i in range(nt-1):
    q[i+1] = A1.solve(-A0.dot(q[i]))

print(np.max(q))
print(np.min(q))

if args.plot:
    import matplotlib.pyplot as plt
    plt.plot(mesh, qinit, label='ic')
    for i in range(args.plot_freq, nt, args.plot_freq):
        plt.plot(mesh, q[i+1], label=str(i))
    plt.legend(loc='center left')
    plt.grid()
    plt.show()
