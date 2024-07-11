import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.fft import fft, ifft
from math import pi
import argparse

# ## === --- --- === ###
#
# Solve the linear advection diffusion equation
#   using sequential timestepping
#
# ADE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: second order central differences
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
    description="Parallel-in-time advection-diffusion equation with implicit theta method",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nt', type=int, default=128, help="Number of timesteps.")
parser.add_argument('--nx', type=int, default=128, help="Number of spatial nodes.")
parser.add_argument('--lx', type=float, default=2*pi, help="Length of domain.")
parser.add_argument('--u', type=float, default=1, help="Advecting velocity.")
parser.add_argument('--re', type=float, default=500, help="Reynolds number.")
parser.add_argument('--cfl', type=float, default=0.8, help="Courant number.")
parser.add_argument('--theta', type=float, default=0.5, help="Implicit parameter.")
parser.add_argument('--alpha', type=float, default=1e-3, help="Circulant parameter.")
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


# Generate block matrices for different coefficients
def block_matrix(l1, l2):
    mat = l1*M + l2*K
    mat.solve = spla.factorized(mat.tocsc())
    return mat


# Build the all-at-once Jacobian

# timestepping matrices
b1col = np.zeros(nt)
b1col[0] = 1/dt
b1col[1] = -1/dt
b1row = np.zeros_like(b1col)
b1row[0] = b1col[0]

b2col = np.zeros(nt)
b2col[0] = theta
b2col[1] = 1-theta
b2row = np.zeros_like(b2col)
b2row[0] = b2col[0]

B1 = linalg.toeplitz(b1col, b1row)
B2 = linalg.toeplitz(b2col, b2row)

# Kronecker Jacobian
A1 = block_matrix(b1col[0], b2col[0])
A0 = block_matrix(b1col[1], b2col[1])
A = spla.aslinearoperator(sparse.kron(B1, M) + sparse.kron(B2, K))


# Circulant preconditioner
class BlockCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, b1, b2, nx, alpha, build_block):
        self.nt = len(b1)
        self.nx = nx
        self.dim = self.nt*self.nx
        self.shape = tuple((self.dim, self.dim))
        self.dtype = b1.dtype

        self.gamma = alpha**(np.arange(self.nt)/self.nt)

        self.eigvals1 = fft(b1*self.gamma, norm='backward')
        self.eigvals2 = fft(b2*self.gamma, norm='backward')
        eigvals = zip(self.eigvals1, self.eigvals2)

        self.blocks = tuple((build_block(l1, l2)
                             for l1, l2 in eigvals))

    def _to_eigvecs(self, v):
        y = np.matmul(np.diag(self.gamma), v)
        return fft(y, axis=0)

    def _from_eigvecs(self, v):
        y = ifft(v, axis=0)
        return np.matmul(np.diag(1/self.gamma), y)

    def _block_solve(self, v):
        for i in range(self.nt):
            v[i] = self.blocks[i].solve(v[i])
        return v

    def _matvec(self, v):
        y = v.reshape((self.nt, self.nx))
        y = self._to_eigvecs(y)
        y = self._block_solve(y)
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real


P = BlockCirculantLinearOperator(b1col, b2col, nx, args.alpha, block_matrix)

# initial conditions
qinit = np.zeros_like(mesh)
qinit[:] = np.cos(mesh/2)**4

# set up timeseries
q = np.zeros(nt*nx)
rhs = np.zeros_like(q)

q = q.reshape((nt, nx))
rhs = rhs.reshape((nt, nx))

# initial guess is constant solution
q[:] = qinit[np.newaxis, :]
rhs[0] -= A0.dot(qinit)

q = q.reshape(nx*nt)
rhs = rhs.reshape(nx*nt)


# residual
def residual(x):
    return rhs - A.matvec(x)


# Krylov solve
niterations = 0


def gmres_callback(pr_norm):
    global niterations
    print(f"niterations: {str(niterations).rjust(5,' ')} | residual: {pr_norm}")
    niterations += 1
    return


q, exit_code = spla.gmres(A, rhs, M=P, x0=q,
                          # restart=nt,
                          # tol=1e-14, atol=1e-14,
                          callback=gmres_callback,
                          callback_type='pr_norm')

print(f"gmres exit code: {exit_code}")
print(f"gmres iterations: {niterations}")
print(f"residual: {linalg.norm(residual(q))}")

# plotting
if args.plot:
    import matplotlib.pyplot as plt
    q = q.reshape((nt, nx))
    plt.plot(mesh, qinit, label='i')
    for i in range(args.plot_freq, nt, args.plot_freq):
        plt.plot(mesh, q[i], label=str(i))
    plt.legend(loc='center left')
    plt.grid()
    plt.show()
