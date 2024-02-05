''' Units and physical constants. '''
import numpy as np

# units
m = 100 # meter [cm]
km = 1000 * m # kilometer [cm]
pc = 3.086e+18 # parsec [cm]
kpc = 1000 * pc # kiloparsec [cm]
Mpc = 1e6 * pc # megaparsec [cm]
hr = 3600 # hour [s]
day = 24 * hr # day [s]
yr = 365 * day # year [s]
Myr = 1e6 * yr # megayear [s]
Gyr = 1e9 * yr # gigayear [s]
ly = 9.463e17 # lightyear [cm]

# physical constants
c = 2.99792458e10 # speed of light [cm/s]
h = 6.6260755e-27 # plank constant [erg s]
hbar = 1.05457266e-27 # reduced plank constant [erg s]
G = 6.67259e-8 # gravitational constant [cm^3/g/s^2]
e = 4.8032068e-10 # electron charge [esu]
m_e = 9.1093897e-28 # electron mass [g]
m_p = 1.6726231e-24 # proton mass [g]
m_n = 1.6749286-24 # neutron mass [g]
m_H = 1.6733e-24 # hydrogen mass [g]
amu = 1.6605402e-24 # atomic mass unit [g]
N_A = 6.0221367e23 # avagadro's number
k_B = 1.380658e-16 # boltzmann constant [erg/K]
eV = 1.6021772e-12 # electron volt [erg]
a = 7.5646e-15 # radiation density constant [erg/cm^3/K^4]
sigma_SB = 5.67051e-5 # stefan-boltzmann constant [erg/cm^2/K^4/s]
alpha = 7.29735308e-3 # fine-structure constant
Ry = 2.1798741e-11 # rydberg constant [erg]

# other useful quantities
M_sol = 1.99e33 # solar mass [g]
R_sol = 6.96e10 # solar radius [cm]
L_sol = 3.9e33 # solar luminosity [erg/s]
T_sol = 5.780e3 # solar temperature [L]
X_sol, Y_sol, Z_sol = 0.7381, 0.2485, 0.0134 # solar abundances
X_cosmo, Y_cosmo, Z_cosmo = 0.7515, 0.2485, 0. # cosmological abundances
temp_HII = 1e4 # hydrogen ionization temperature [Kelvin]
energy_HII = 13.59844 * eV # hydrogen ionization energy [erg]

# cosmology (from Plank https://arxiv.org/abs/1502.01589)
h0 = 0.703 # normalized Hubble constant
H0 = h0 * 100 * km / Mpc # Hubble constant, adjusted for consistency with RAMSES [1/s]
Omega_m0 = 0.276 # Matter density parameter, adjusted for consistency with RAMSES
Omega_b0 = 0.049 # Baryon density parameter, adjusted for consistency with RAMSES
Omega_L0 = 0.724 # Dark energy density parameter, adjusted for consistency with RAMSES
Omega_k0 = 0.298e-7 # Curvature density parameter, adjusted for consistency with RAMSES
rho_crit_0 = 3 * H0**2 / (8 * np.pi * G) # critical density [g/cm^3]
sigma8 = 0.811 # RMS matter fluctuations averaged over an R = 8h^(-1) sphere
n_PS = 0.9667 # power law of the initial power spectru,
