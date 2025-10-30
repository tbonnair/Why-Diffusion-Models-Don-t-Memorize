"""

----------------------------------------------------------------

This script reproduces the spectral density plots of the paper:

  "Why Diffusion Models Don’t Memorize: The Role of
Implicit Dynamical Regularization in Training"

Author: Raphael Urfin
Contact: raphael.urfin@phys.ens.fr


Description:
------------
Solves the Saddle point equations for the Stieljes of U, computes the eigenvalue density rho and compare it to the empirical histogram of eigenvalues of U.
This code is for Gaussian data with covariance Sigma with spectral density rho_Sigma(\lambda)=0.5\delta(\lambda-\lambda_1)+0.5\delta(\lambda-\lambda_2).
To reproduce the plots of the paper you can take lambda1=lambda2=1 or lambda1=0.5 and lambda2=1.5.

All key parameters (psi_p, psi_n, d, t, lambda1, lambda2) can be passed via command-line arguments.

Example usage:
--------------
    python src/main.py --psi_p 2.0 --psi_n 1.0 --lambda1 1.0 --lambda2 2.0 --d 2000 --t 0.01
"""


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight']= 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300      
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits']=(-6, 6)
mpl.rcParams['axes.formatter.use_mathtext']=True

#mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# %%
cmap = mpl.colormaps['Dark2']
colors = cmap.colors
# -------------------------------
# Argument parser
# -------------------------------
parser = argparse.ArgumentParser(description="Compute and plot spectral density of random feature diffusion model.")

parser.add_argument("--lambda1", type=float, default=1.0, help="First eigenvalue of Sigma (default: 0.5)")
parser.add_argument("--lambda2", type=float, default=2.0, help="Second eigenvalue of Sigma (default: 1.5)")
parser.add_argument("--psi_p", type=float, default=2.0, help="Aspect ratio psi_p = p/d (default: 8.0)")
parser.add_argument("--psi_n", type=float, default=1.0, help="Aspect ratio psi_n = n/d (default: 4.0)")
parser.add_argument("--d", type=int, default=int(2e3), help="Feature dimension (default: 2000)")
parser.add_argument("--t", type=float, default=1e-2, help="Diffusion time parameter (default: 1e-2)")

args = parser.parse_args()

lambda1 = args.lambda1
lambda2 = args.lambda2
psi_p = args.psi_p
psi_n = args.psi_n
d = args.d
t = args.t

# -------------------------
# Model / solver parameters (change as needed)
# -------------------------




# We will build Sigma as diagonal with half entries = lambda1, half = lambda2
# (dimension d). If d is odd the extra eigenvalue is lambda1.
eigvals_Sigma = np.array([lambda1]*(d//2) + [lambda2]*(d - d//2))
Sigma = np.diag(eigvals_Sigma)

#computes the constants that appear in the equations
def sigma_fun(x):
    return np.tanh(x)

mc_samples = 500_000  
sigma_xv = np.trace(Sigma) / float(d)    # sigma_{\vx}
Delta_t = 1.0 - np.exp(-2.0 * t)
Gamma_t_sq = (np.exp(-2.0 * t) * sigma_xv**2) + Delta_t
Gamma_t = np.sqrt(Gamma_t_sq)


rng = np.random.default_rng(12345)

u = rng.standard_normal(mc_samples)
v = rng.standard_normal(mc_samples)
w = rng.standard_normal(mc_samples)
z = rng.standard_normal(mc_samples)


coeff1 = np.exp(-t) * sigma_xv
sqrtD = np.sqrt(max(Delta_t, 0.0))
A_uv = coeff1 * u + sqrtD * v
A_uw = coeff1 * u + sqrtD * w  # for the second factor in v_t^2 expectation

# sigma values
sigma_A_uv = sigma_fun(A_uv)
sigma_A_uw = sigma_fun(A_uw)

# a_t: E[ sigma(A) * (u / (e^{-t} sigma_xv)) ]
eps = 1e-16
denom_for_a = coeff1 if abs(coeff1) > eps else eps   # avoid div by zero
a_t = np.mean(sigma_A_uv * (u / denom_for_a))

# b_t: E[ v * sigma(A) ]
b_t = np.mean(v * sigma_A_uv)

# v_t^2: E_{u,v,w}[ sigma(A(u,v)) * sigma(A(u,w)) ] - a_t^2 * e^{-2t} * sigma_xv^2
first_term = np.mean(sigma_A_uv * sigma_A_uw)
v_t_sq = first_term - (a_t**2) * (np.exp(-2.0*t) * sigma_xv**2)
v_t=np.sqrt(v_t_sq)

#
# s_t^2: E_z[ sigma(Gamma_t * z)^2 ] - a_t^2 e^{-2t} sigma_xv^2 - v_t^2 - b_t^2
sigma_Gz = sigma_fun(Gamma_t * z)
s_t_sq = np.mean(sigma_Gz**2) - (a_t**2) * (np.exp(-2.0*t) * sigma_xv**2) - v_t_sq - (b_t**2)



# -------------------------
# Put params into dict (for solver)
# -------------------------
params = {
    'a_t': a_t, 'b_t': b_t, 'v_t': v_t, 't': t,
    'p': p, 'n': n, 'psi_p': psi_p, 's_t_sq': s_t_sq,
    'lambda1': lambda1, 'lambda2': lambda2
}



# -------------------------
# Theoretical solver functions (6 real eqns for 3 complex unknowns)
# -------------------------
def hat_s_of_q(q, b_t, psi_p):
    return b_t**2 * psi_p + 1.0 / q

def hat_r_of_rq(r, q, a_t, t, p, n, psi_p, v_t):
    denom = 1.0 + (a_t**2 * np.exp(-2.0*t) * p / n) * r + (p * v_t**2 / n) * q
    return (psi_p * a_t**2 * np.exp(-2.0*t)) / denom

def equations(vars, lambda_val, epsilon, params):
    q_re, q_im, r_re, r_im, s_re, s_im = vars
    q = q_re + 1j * q_im
    r = r_re + 1j * r_im
    s = s_re + 1j * s_im
    z = lambda_val - 1j * epsilon

    a_t = params['a_t']; b_t = params['b_t']; v_t = params['v_t']; t = params['t']
    p = params['p']; n = params['n']; psi_p = params['psi_p']; s_t_sq = params['s_t_sq']
    l1 = params['lambda1']; l2 = params['lambda2']

    hs = hat_s_of_q(q, b_t, psi_p)
    hr = hat_r_of_rq(r, q, a_t, t, p, n, psi_p, v_t)

    denom1 = hs + l1 * hr
    denom2 = hs + l2 * hr

    s_calc = 0.5 * (1.0 / denom1 + 1.0 / denom2)
    r_calc = 0.5 * (l1 / denom1 + l2 / denom2)

    denom_common = 1.0 + (a_t**2 * np.exp(-2.0*t) * p / n) * r + (p * v_t**2 / n) * q
    third = psi_p * (s_t_sq - z) + (psi_p * v_t**2) / denom_common + (1.0 - psi_p) / q - s / (q**2)

    eqs = np.empty(6)
    eqs[0] = (s - s_calc).real
    eqs[1] = (s - s_calc).imag
    eqs[2] = (r - r_calc).real
    eqs[3] = (r - r_calc).imag
    eqs[4] = third.real
    eqs[5] = third.imag

    return eqs


def initialization_q(lambda_i, params):
    A=100
    z=lambda_i+1j*(A)
    q0 = 1.0 / z
    r0 = q0
    s0 = q0
    q_r, q_im,r_r, r_im,s_r,s_im=q0.real, q0.imag,r0.real, r0.imag,s0.real,s0.imag
    solution = root(equations, [ q_r, q_im,r_r, r_im,s_r,s_im], args=(z.real, A, params), method='lm').x
    for k in range(A,0,-1):
        solution = root(equations, solution, args=(z.real, k,  params), method='lm').x
    for k in [0.9,0.7,0.5,0.3,0.1,0.01,0.001,0.0001,0.00001,1e-6,1e-7,1e-8,1e-9,]:
        solution = root(equations, solution, args=(z.real, k,  params), method='lm').x
    return solution



def solve_equations(lambda_val,initialization,params):
    return root( equations, initialization, args=(lambda_val,0,params), method='lm').x

#range of lambda on which you solve the equations
lambda_values = np.linspace(-0.1, 60, 300)   # for second bulk
#lambda_values = np.linspace(-0.05, 0.5, 300) #for first bulk
lambda_values = np.concatenate((np.linspace(-0.1, 0.5, 100),  np.linspace(0.5, 60, 300)))
imag_q_values = np.zeros(len(lambda_values))

for i in range(len(lambda_values)):
  initial_guess=initialization_q(lambda_values[i], params)
  q_solution =initial_guess[1]
  imag_q_values[i] = (abs(q_solution) / np.pi)


mask1 = lambda_values < 1.
mask2 = lambda_values >= 1.

# Plot each part with different colors
plt.plot(lambda_values[mask1], imag_q_values[mask1],
         color=colors[2], label='theoretical $\\rho(\\lambda)$', linewidth=2)
plt.plot(lambda_values[mask2], imag_q_values[mask2],
         color=colors[1], linewidth=2)


# -------------------------
# Gaussian equivalent U
# -------------------------

rng3 = np.random.default_rng(20241014)

# W as before (p x d)
W = rng3.standard_normal((p, d)) / np.sqrt(d)

# X' ~ N(0, Sigma)
Z = rng3.standard_normal((d, n))
sqrt_eigs = np.sqrt(eigvals_Sigma)
Xprime = sqrt_eigs[:, None] * Z

# Omega ~ N(0,1), same shape as G (p x n)
Omega = rng3.standard_normal((p, n))

# Build G according to lemma
G = np.exp(-t) * a_t * (W @ Xprime) + v_t * Omega

# Build U = 1/n G G^T + b_t^2 W W^T / d + s_t^2 I_p
U = (G @ G.T) / n + b_t**2 * (W @ W.T) / d + s_t_sq * np.eye(p)

# Compute eigenvalues
eigvals_U, eigvecs_U = eigh(U)  # returns tuple (eigvals, eigvecs)
eigvals_U.sort()                 # now sort works
eigvals_U = eigvals_U[eigvals_U < np.max(lambda_values)]
# Histogram / density

hist_bins = 50 #for the second bulk


counts, bin_edges = np.histogram(eigvals_U, bins=hist_bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


#plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.35, color=colors[b], label='empirical Gaussian equiv', edgecolor='none')
bar_colors = [colors[2] if c < 1. else colors[1] for c in bin_centers]

plt.bar(bin_centers,
        counts,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.35,
        color=bar_colors,
        label='empirical Gaussian equiv',
        edgecolor='none')



plt.ylim(0, 0.008)
plt.xlim(np.min(lambda_values), np.max(lambda_values))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
filename = f"results/Plot_spectrum_psi_p={psi_p:.1f}_psi_n={psi_n:.1f}_t={t:.0e}.pdf"
plt.tight_layout()
plt.savefig(filename, bbox_inches='tight')
print(f"✅ Plot saved to: {filename}")
plt.show()
