''' Units and physical constants. '''
import numpy as np

# units
AA = 1e-8 # Angstroms [cm]
micron = 1e-4 # micron [cm]
m = 100 # meter [cm]
km = 1000 * m # kilometer [cm]
pc = 3.086e+18 # parsec [cm]
kpc = 1000 * pc # kiloparsec [cm]
Mpc = 1e6 * pc # megaparsec [cm]
hr = 3600 # hour [s]
day = 24 * hr # day [s]
wk = 7 * day # week [s]
yr = 365 * day # year [s]
kyr = 1e3 * yr # kiloyear [s]
Myr = 1e6 * yr # megayear [s]
Gyr = 1e9 * yr # gigayear [s]
ly = 9.463e17 # lightyear [cm]

# physical constants
c = 2.9979246e+10 # speed of light [cm/s]
h = 6.6260702e-27 # plank constant [erg s]
hbar = h / (2*np.pi) # reduced plank constant [erg s]
G = 6.67408e-08 # gravitational constant [cm^3/g/s^2]
e = 4.8032068e-10 # electron charge [esu]
m_e = 9.1093897e-28 # electron mass [g]
m_p = 1.6726231e-24 # proton mass [g]
m_n = 1.6749286-24 # neutron mass [g]
m_H = 1.660539e-24 # hydrogen mass [g]
amu = 1.6605402e-24 # atomic mass unit [g]
N_A = 6.0221367e23 # avagadro's number
k_B = 1.3806490e-16 # boltzmann constant [erg/K]
eV = 1.6021766e-12 # electron volt [erg]
a_rad = 7.5657233e-15 # radiation density constant [erg/cm^3/K^4]
sigma_SB = 5.67051e-5 # stefan-boltzmann constant [erg/cm^2/K^4/s]
alpha = 7.29735308e-3 # fine-structure constant
Ry = 2.1798741e-11 # rydberg constant [erg]
sigma_T = 6.6524587e-25 # Thompson scattering cross section [cm^2]

# other useful quantities
M_sol = 1.9891e+33 # solar mass [g]
R_sol = 6.96e10 # solar radius [cm]
L_sol = 3.828e+33 # solar luminosity [erg/s]
T_sol = 5.780e3 # solar temperature [L]
X_sol, Y_sol, Z_sol = 0.7381, 0.2485, 0.0134 # solar abundances
X_cosmo, Y_cosmo, Z_cosmo = 0.76, 0.24, 0. # cosmological abundances
temp_HII = 1e4 # hydrogen ionization temperature [Kelvin]
energy_HII = 13.59844 * eV # hydrogen ionization energy [erg]

# cosmology (from RAMSES)
h0 = 0.703 # normalized Hubble constant [1/s]
H0 = h0 * 100 * km / Mpc # Hubble constant [1/s]
Omega_m0 = 0.276 # Matter density parameter
Omega_b0 = 0.049 # Baryon density parameter
Omega_L0 = 0.724 # Dark energy density parameter
Omega_k0 = 0.298e-7 # Curvature density parameter
rho_crit_0 = 3 * H0**2 / (8 * np.pi * G) # critical density [g/cm^3]

# cosmology (from Plank https://arxiv.org/abs/1502.01589)
sigma8 = 0.811 # RMS matter fluctuations averaged over an R = 8h^(-1) sphere 
n_PS = 0.9667 # power law of the initial power spectrum

# numerical constants
small = 1e-32
