import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from numba import njit
# ---------------------- Utility: complex integration -----------------------
def quad_complex(func, a, b, **kwargs):
    """Integrate a complex-valued function by integrating real and imag parts."""
    real = lambda x: np.real(func(x))
    imag = lambda x: np.imag(func(x))
    R, _ = quad(real, a, b, **kwargs)
    I, _ = quad(imag, a, b, **kwargs)
    return R + 1j * I

n = 100
xi, wi = np.polynomial.legendre.leggauss(n)

#@njit
def fast_gauss_integrate(f, a, b):
    # Scale nodes
    x_scaled = 0.5*(b - a)*xi + 0.5*(b + a)
    w_scaled = 0.5*(b - a)*wi
    s = 0.0
    for i in range(len(xi)):
        s += w_scaled[i] * f(x_scaled[i])
    return s


def integrand_G0(xi, zm, zn):
    return basis(xi, zn) * kern(zm, xi)

def integrand_Gp(xi, zm, zn, h):
    return basis(xi, zn) * kern(zm, xi - h)

def integrand_Gm(xi, zm, zn, h):
    return basis(xi, zn) * kern(zm, xi + h)


def triangle(x, n, N, L):
    pulse_length = L / N
    a = n * pulse_length
    b = (n + 1) * pulse_length
    mid = a + 0.5 * (b - a)    
    if a <= x < mid:
        return (x - a) * 2 / (b - a)        
    elif mid <= x <= b:
        return 1 - (x - mid) * 2 / (b - a)  
    else:
        return 0.0

def gauss_quadrature(a,b):
    n = 100
    xi, wi = np.polynomial.legendre.leggauss(n)
    xi_scaled = 0.5 * (b - a) * xi + 0.5 * (b + a)
    wi_scaled = 0.5 * (b - a) * wi
    final = [(xi_scaled[i], wi_scaled[i]) for i in range(0,n)]
    return final

# ---------------------- Physical constants & geometry ----------------------
f = 300e6                    # Hz
c = 3e8                      # m/s
lam = c / f                  # wavelength
k = 2 * np.pi / lam          # wavenumber
mu0 = 4 * np.pi * 1e-7       # H/m
eta0 = 377.0                 # Ohms
L = lam / 2                  # dipole length (half-wave)

# One-time before loops
Ng = 12  # (not used directly here; using quad integrator instead)
# helper mapping (if you want to use Gauss-Legendre nodes later)
def map_interval(a_, b_, x):
    return 0.5 * (b_ - a_) * x + 0.5 * (a_ + b_)
def jac(a_, b_):
    return 0.5 * (b_ - a_)

# Green's function and derivatives (expressions from Balanis)
def Gfun(R):
    return np.exp(-1j * k * R) / (4 * np.pi * R)

def dGdR(R):
    return np.exp(-1j * k * R) * (-1 / R**2 - 1j * k / R) / (4 * np.pi)

def d2GdR2(R):
    return np.exp(-1j * k * R) * (2 / R**3 + 2j * k / R**2 - (k**2) / R) / (4 * np.pi)

def d2Gdz2(Delta, R, a):
    return ((Delta / R)**2) * d2GdR2(R) + (a**2 / R**3) * dGdR(R)

# ---------------------- Sweep of N and MoM solve ----------------------------
NN = np.linspace(21, 201, 7, dtype=int)   # same as MATLAB
Zin_ar = np.zeros_like(NN, dtype=complex)

for idx, N in enumerate(NN):
    print(f"Working on N = {N}")
    dz = L / N
    a = min(lam / 800.0, 0.15 * dz)
    z = np.linspace(-L / 2 + dz / 2, L / 2 - dz / 2, N)  # segment centers
    h = dz / 20.0

    # Delta-gap excitation (center)
    V = np.zeros(N, dtype=complex)
    mid_index = N // 2
    V[mid_index] = 1.0 + 0j

    # Initialize Z
    Z = np.zeros((N, N), dtype=complex)

    # Define basis and kernel (sinusoidal basis like MATLAB)
    def basis(zp, zn):
        return np.sin(k * (L / 2 - np.abs(zp - zn)))

    def kern(zm, zp):
        R = np.sqrt((zm - zp)**2 + a**2)
        return np.exp(-1j * k * R) / R

    # Assemble Z (double loop)
    for m in range(N):
        zm = z[m]
        for n in range(N):
            zn = z[n]

            zp1 = zn - dz / 2.0
            zp2 = zn + dz / 2.0

            # integrands (complex)
            f_G0 = lambda zp: basis(zp, zn) * kern(zm, zp)
            f_Gp = lambda zp: basis(zp, zn) * kern(zm, zp - h)
            f_Gm = lambda zp: basis(zp, zn) * kern(zm, zp + h)

            # integrate complex functions by integrating real and imag parts
            # increase limit to handle difficult integrands
            # points_weights = gauss_quadrature(zp1, zp2)
            # G0 = sum([basis(xi, zn) * kern(zm, xi) * wi for xi, wi in points_weights])
            # Gp = sum([basis(xi, zn) * kern(zm, xi - h) * wi for xi, wi in points_weights])
            # Gm = sum([basis(xi, zn) * kern(zm, xi + h) * wi for xi, wi in points_weights])

            G0 = quad_complex(f_G0, zp1, zp2, limit=200)
            Gp = quad_complex(f_Gp, zp1, zp2, limit=200)
            Gm = quad_complex(f_Gm, zp1, zp2, limit=200)

            #G0 = fast_gauss_integrate()
            # a = zp1
            # b = zp2
            # G0 = fast_gauss_integrate(lambda xi: integrand_G0(xi, zm, zn), a, b)
            # Gp = fast_gauss_integrate(lambda xi: integrand_Gp(xi, zm, zn, h), a, b)
            # Gm = fast_gauss_integrate(lambda xi: integrand_Gm(xi, zm, zn, h), a, b)

            d2G = (Gp - 2 * G0 + Gm) / (h**2)

            # Z element (same prefactor used in your MATLAB code)
            Z[m, n] = (1j * eta0 * dz / (4 * np.pi * k)) * (d2G + (k**2) * G0)

    # Solve for currents
    I = np.linalg.solve(Z, V)

    Zin = 1.0 / I[mid_index]
    Zin_ar[idx] = Zin
    print(f"  Zin (center) = {Zin.real:.4f} {Zin.imag:+.4f}j")

# ---------------------- Plot convergence (real part of Zin) ----------------
plt.figure()
plt.plot(NN, Zin_ar.real, 'b-', linewidth=2)
plt.xlabel('No. of sections N')
plt.ylabel('Re(Zin) [Ohm]')
plt.title('Convergence (Pocklington via Balanis) — Re(Zin)')
plt.grid(True)

# ---------------------- Far-field pattern for last computed N ------------------
# Use the last I and z from the final N loop
# (If you want patterns for each N, you can compute inside the loop)
theta = np.linspace(0, np.pi, 500)
E_theta = np.zeros_like(theta, dtype=float)

for i, th in enumerate(theta):
    sum_val = 0+0j
    for n in range(N):
        sum_val += I[n] * np.exp(1j * k * z[n] * np.cos(th)) * dz
    E_theta[i] = np.abs(sum_val) * np.sin(th)

# Normalize
E_theta /= np.max(E_theta)

# Plot radiation pattern (linear) and in dB on polar
plt.figure()
plt.plot(theta * 180 / np.pi, 20 * np.log10(np.clip(E_theta, 1e-12, None)), '-k', linewidth=1.5)
plt.xlabel('Theta (deg)')
plt.ylabel('20 log10 |E_theta| (dB)')
plt.title('Radiation pattern (cut)')
plt.grid(True)

# Polar dB plot similar to MATLAB's PolardB
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='polar')
# Convert theta so 0 at top and clockwise:
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.plot(theta, 20 * np.log10(np.clip(E_theta, 1e-12, None)), '-k', linewidth=1.2)
ax.set_ylim(-25, 0)
ax.set_title(r"$20\log_{10}|E_{\theta}|$ [dB]")

# ---------------------- Sanity-check pattern vs half-sine analytic ------------
# dense theta for accuracy
theta_dense = np.linspace(0, np.pi, 2001)
F_mom = np.array([np.abs(np.sum(I * np.exp(1j * k * z * np.cos(th))) * dz) * np.sin(th) for th in theta_dense])
Iref = np.sin(k * (L / 2 - np.abs(z)))
F_ref = np.array([np.abs(np.sum(Iref * np.exp(1j * k * z * np.cos(th))) * dz) * np.sin(th) for th in theta_dense])

F_mom /= np.max(F_mom)
F_ref /= np.max(F_ref)

im_peak = np.argmax(F_mom)
ir_peak = np.argmax(F_ref)
theta_peak_mom_deg = theta_dense[im_peak] * 180 / np.pi
theta_peak_ref_deg = theta_dense[ir_peak] * 180 / np.pi

print(f"Peak angle (MoM)      : {theta_peak_mom_deg:.2f} deg")
print(f"Peak angle (analytic) : {theta_peak_ref_deg:.2f} deg")

# Polar comparison
fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='polar')
ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
ax2.plot(theta_dense, F_mom, linewidth=1.6, label='MoM')
ax2.plot(theta_dense, F_ref, '--', linewidth=1.6, label='Analytic half-sine')
ax2.legend(loc='upper right')
ax2.set_title('Dipole pattern sanity check (main lobe at ~90°)')

# ---------------------- Power balance & input impedance check ----------------
Nt = 2001
theta_ff = np.linspace(0, np.pi, Nt)
Prad_integrand = np.zeros(Nt, dtype=float)
sinth = np.sin(theta_ff)

for it in range(Nt):
    Fth = np.sum(I * np.exp(1j * k * z * np.cos(theta_ff[it]))) * dz
    Prad_integrand[it] = (sinth[it]**3) * np.abs(Fth)**2

Prad = (eta0 / 2) * (k**2 / (8 * np.pi)) * np.trapz(Prad_integrand, theta_ff)

Ifeed = I[mid_index]
V0 = 1.0
Pin = 0.5 * np.real(V0 * np.conj(Ifeed))
Zin_mom = 1.0 / Ifeed

print("\n---- Power & Impedance Check ----")
print(f"Zin (MoM)     = {Zin_mom.real:.2f} {Zin_mom.imag:+.2f}j ohms")
print(f"Pin (from feed) = {Pin:.6g} W")
print(f"Prad (from FF)  = {Prad:.6g} W")
print(f"Prad/Pin        = {Prad / Pin:.3f}")
print("Reference Zin (half-wave dipole) ~ 73 + j0 ohms")
print(f"Zin error vs 73Ω = {100*(Zin_mom.real - 73)/73:.2f} % (real part)")

# ---------------------- Plot current distribution ---------------------------
plt.figure()
plt.plot(z, np.abs(I), 'b-', linewidth=2)
plt.xlabel('z (m)')
plt.ylabel('|I(z)| (A)')
plt.title('Current Distribution on Wire (Pocklington via Balanis)')
plt.grid(True)
plt.show()
