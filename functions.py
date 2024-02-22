import os, subprocess
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import erf
import const
from collections.abc import Iterable
from config import *

# simple functions for curve fitting
linear = lambda x, a: a*x
affine = lambda x, a, b: a + b*x
quadratic = lambda x, a, b, c: a + b*x + c*x**2
cubic = lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3
powerlaw = lambda x, a, b: a*x**b
def broken_powerlaw(x, con, x_break, exp1, exp2, sharpness):
    return con * (x / x_break)**exp1 * (1 + (x / x_break)**(np.abs(exp2) * sharpness))**(np.sign(exp2) / sharpness)

class Terminal:
    ''' Context manager for running commands in the terminal from Python. '''
    def __enter__(self):
        self.cwd = os.getcwd()
        return None
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self.cwd)

def qtile_weighted(field, weight, qtile):
    ''' Get the weighted q-tile of a distribution. '''
    idx_sorted = np.argsort(field)
    cumsum = np.cumsum(weight[idx_sorted])
    field_median = field[idx_sorted[np.searchsorted(cumsum, qtile * cumsum[-1])]]
    return field_median

def median_weighted(field, weight):
    ''' Get the weighted median of a distribution. '''
    return qtile_weighted(field, weight, 0.5)

def get_stdout(cmd):
    ''' Return the standard output of a command line directive '''
    return subprocess.check_output(cmd, shell=True).decode()

def clear_figures():
    ''' Move all current figures to legacy folder. '''
    with Terminal() as terminal:
        os.chdir(save_dir)
        os.system("mv *.png ../legacy/")

def git_commit(git_message=None):
    ''' Commit relevant files to git and push changes. '''
    with Terminal() as terminal:
        
        os.chdir(analysis_dir)
        if git_message == None: git_message = "pushing updates to analysis code"
        os.system("git add *.py")
        os.system("git add *.ipynb")
        os.system("git commit -m '%s'" % git_message)
        os.system("git push")

def save_fig(fig_name, filetype="png", dpi=300):
    '''
    Save the current matplotlib figure.

    Args
    name (string): figure name
    filetype (string): file type
    dpi (int): dots per inch
    '''
    datetime_string = datetime.now().strftime("%m%d%Y%H%M")
    filename = "%s-%s.%s" % (fig_name, datetime_string, filetype)
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight", dpi=dpi)
    print("Saved figure as '%s'" % filename)

def norm(A):
    ''' Compute the norm of a vector field. '''
    return np.sqrt(np.sum(A**2, axis=0))

def dot(A, B):
    ''' Compute the dot product of two vector fields. '''
    return np.sum(A * B, axis=0)

def proj(A, B):
    ''' Compute the projection of one vector field onto another. '''
    return (dot(A, B) / norm(B)**2)[None, :, :, :] * B

def calc_eps_sf(alpha_vir, mach_turb, b_turb=1.0, eps_sf_loc=1.0):
    ''' 
    Star formation efficiency in the multi-freefall model.
    See Federrath&Klessen2012 (https://arxiv.org/pdf/1209.2856.pdf) and Kretschmer&Teyssier2021 (https://arxiv.org/pdf/1906.11836.pdf) for details.

    Args
    alpha_vir: virial parameter
    mach_turb: turbulent Mach number
    b_turb: turbulence forcing parameter
        varies smoothly between b ~ 1/3 for purely solenoidal (divergence-free) forcing 
        and b ~ 1 for purely compressive (curl-free) forcing
    eps_sf_loc: SFE on the sonic scale

    Returns
    eps_sf: star formation efficiency
    '''
    s_crit = np.log(alpha_vir * (1 + 2 * mach_turb**4 / (1 + mach_turb**2))) # lognormal critical density for star formation
    sigma_s = np.sqrt(np.log(1 + b_turb**2 * mach_turb**2)) # standard deviation of the lognormal subgrid density distribution
    eps_sf = eps_sf_loc/2 * np.exp(3/8 * sigma_s**2) * (1 + erf((sigma_s**2 - s_crit) / np.sqrt(2 * sigma_s**2))) # star formation efficiency
    return eps_sf
    
def calc_eps_sf2(density, energy_turb, temp, dx, b_turb=1.0, eps_sf_loc=1.0):
    ''' 
    Wrapper for calc_eps_sf, in terms of more basic quantities.

    Args
    density: density
    energy_turb: turbulent energy
    temp: temperature
    dx: resolution
    b_turb: turbulence forcing parameter
        varies smoothly between b ~ 1/3 for purely solenoidal (divergence-free) forcing 
        and b ~ 1 for purely compressive (curl-free) forcing
    eps_sf_loc: SFE on the sonic scale

    Returns
    eps_sf: star formation efficiency
    '''
    c_s = np.sqrt(const.k_B * temp / const.m_p) # sound speed
    mach_turb = np.sqrt(2/3 * energy_turb) / c_s # turbulent Mach number
    alpha_vir = 15 / np.pi * c_s**2 * (1 + mach_turb**2) / (const.G * density * dx**2) # virial parameter
    return calc_eps_sf(alpha_vir, mach_turb, b_turb=b_turb, eps_sf_loc=eps_sf_loc)

def aexp_to_proper_time(a, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert expansion factor to proper time.'''
    integrand = lambda a: (Omega_m0 * a**(-1) + Omega_k0 + Omega_L0 * a**2)**(-1/2)
    t = quad(integrand, 0, a)[0] / H0
    return t

def aexp_to_conformal_time(a, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert expansion factor to conformal time.'''
    integrand = lambda a: (Omega_m0 * a + Omega_k0 * a**2 + Omega_L0 * a**4)**(-1/2)
    tau = const.c * quad(integrand, 0, a)[0] / H0
    return tau

def proper_time_to_aexp(t, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert proper time to expansion rate.'''
    a = fsolve(lambda a: (aexp_to_proper_time(a, Omega_m0, Omega_k0, Omega_L0, H0) - t) * H0, 0.1)
    return a

def conformal_time_to_aexp(tau, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert conformal time to expansion rate.'''
    a = fsolve(lambda a: (aexp_to_conformal_time(a, Omega_m0, Omega_k0, Omega_L0, H0) - tau) * H0, 0.1)
    return a

def symlog(x, C=1):
    ''' Compute the symmetric log '''
    return np.sign(x) * np.log10(1 + np.abs(x / C))

def solve_quadratic(A, B, C):
    ''' 
    Solve a quadratic equation. 
    
    Note that we choose between forms of the quadratic formula to maximize numerical stability.
    See https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very/2007723#2007723
    '''
    assert np.shape(A) == np.shape(B), "A and B must have the same shape."
    assert np.shape(B) == np.shape(C), "B and C must have the same shape."
    
    x1, x2 = np.zeros_like(B), np.zeros_like(B)
    x1[B != 0] = ((-B - np.sign(B) * np.sqrt(B**2 - 4 * A * C)) / (2 * A))[B != 0]
    x2[B != 0] = C[B != 0] / (A[B != 0] * x1[B != 0])
    return x1, x2

def calc_AH_coords(h, ph):
    ''' 
    Compute the Aitoff-Hammer projection coordinates.
    
    Args
    h: polar coordinate
    ph: azimuthal coordinate
    
    Returns
    AH1, AH2: Aitoff-Hammer projection coordinates
    '''
    h_bar = np.pi/2 - h
    lamb = ph - np.pi
    AH1 = 2 * np.cos(h_bar) * np.sin(lamb / 2) / np.sqrt(1 + np.cos(h_bar) * np.cos(lamb / 2))
    AH2 = np.sin(h_bar) / np.sqrt(1 + np.cos(h_bar) * np.cos(lamb / 2))
    return AH1, AH2

def coord_conv(coord, sys1, sys2):
    '''
    Convert a coordinate array from one coordinate system to another.
    See Wikipedia (https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates) for formulas.
    
    Args
    coord: coordinate array
    sys1 (int): coordinate system of coord
    sys2 (int): new coordinate array
    
    Returns
    coord_new: coordinate array in the new coordinate system
    '''
    assert sys1 in [CART, CYL, SPH], "Argmument sys1 is not a valid coordinate system."
    assert sys2 in [CART, CYL, SPH], "Argmument sys2 is not a valid coordinate system."
    coord_new = np.zeros_like(coord)
    if sys2 == sys1:
        coord_new = np.copy(coord)
    elif sys1 == CART and sys2 == SPH:
        coord_new[R] = np.sqrt(np.sum(coord**2, axis=0))
        coord_new[H] = np.arctan2(np.sqrt(coord[X]**2 + coord[Y]**2), coord[Z])
        coord_new[PH] = np.arctan2(coord[Y], coord[X])
    elif sys1 == CART and sys2 == CYL:
        coord_new[S] = np.sqrt(coord[X]**2 + coord[Y]**2),
        coord_new[PH2] = np.arctan2(coord[Y], coord[X])
        coord_new[Z] = coord[Z]
    elif sys1 == SPH and sys2 == CART:
        coord_new[X] = coord[R] * np.cos(coord[PH]) * np.sin(coord[H])
        coord_new[Y] = coord[R] * np.sin(coord[PH]) * np.sin(coord[H])
        coord_new[Z] = coord[R] * np.cos(coord[H])
    elif sys1 == SPH and sys2 == CYL:
        coord_new[S] = coord[R] * np.sin(coord[H])
        coord_new[PH2] = coord[PH]
        coord_new[Z] = coord[R] * np.cos(coord[H])
    elif sys1 == CYL and sys2 == CART:
        coord_new[X] = coord[R] * np.cos(coord[PH2])
        coord_new[Y] = coord[R] * np.sin(coord[PH2])
        coord_new[Z] = coord[Z]
    elif sys1 == CYL and sys2 == SPH:
        coord_new[R] = np.sqrt(coord[S]**2 + coord[Z]**2)
        coord_new[H] = np.arctan2(coord[S] / coord[Z])
        coord_new[PH] = coord[PH2]
    return coord_new

def vec_conv(field, coord, sys1, sys2):
    '''
    Convert a vector gield array from one coordinate system to another.
    See Wikipedia (https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates) for formulas.
    
    Args
    field: field
    coord: coordinate array
    sys1 (int): coordinate system of coord and field
    sys2 (int): new coordinate array
    
    Returns
    coord_new: coordinate array in the new coordinate system
    '''    
    assert sys1 in [CART, CYL, SPH], "Argmument sys1 is not a valid coordinate system."
    assert sys2 in [CART, CYL, SPH], "Argmument sys2 is not a valid coordinate system."
    field_new = np.zeros_like(field)
    if sys1 == sys2:
        field_new = np.copy(field)
    elif sys1 == CART and sys2 == SPH:
        coord_s = np.sqrt(coord[X]**2 + coord[Y]**2)
        coord_r = np.sqrt(np.sum(coord**2, axis=0))
        field_new[R] = np.sum(coord * field, axis=0) / coord_r
        field_new[H] = ((coord[X] * field[X] + coord[Y] * field[Y]) * coord[Z] - coord_s**2 * field[Z]) / (coord_r * coord_s)
        field_new[PH] = (-coord[Y] * field[X] + coord[X] * field[Y]) / coord_s 
    elif sys1 == CART and sys2 == CYL:
        coord_s = np.sqrt(coord[X]**2 + coord[Y]**2)
        field_new[S] = (coord[X] * field[X] + coord[Y] * field[Y]) / coord_s
        field_new[PH2] = (-coord[Y] * field[X] + coord[X] * field[Y]) / coord_s
        field_new[Z] = field[Z]
    elif sys1 == SPH and sys2 == CART:
        field_new[X] = np.sin(coord[H]) * np.cos(coord[PH]) * field[R] + np.cos(coord[H]) * np.cos(coord[PH]) * field[H] - np.sin(coord[PH]) * field[PH]
        field_new[Y] = np.sin(coord[H]) * np.sin(coord[PH]) * field[R] + np.cos(coord[H]) * np.sin(coord[PH]) * field[H] - np.cos(coord[PH]) * field[PH]
        field_new[Z] = np.cos(coord[H]) * field[R] - np.sin(coord[PH]) * field[H]
    elif sys1 == SPH and sys2 == CYL:
        field_new[S] = np.sin(coord[H]) * field[R] + np.cos(coord[H]) * field[H]
        field_new[PH2] = field[PH]
        field_new[Z] = np.cos(coord[H]) * field[R] - np.sin(coord[H]) * field[H]
    elif sys1 == CYL and sys2 == CART:
        field_new[X] = np.cos(field[PH2]) * field[S] - np.sin(coord[PH2]) * field[PH2]
        field_new[Y] = np.sin(field[PH2]) * field[S] + np.cos(coord[PH2]) * field[PH2]
        field_new[Z] = field[Z]
    elif sys1 == CYL and sys2 == SPH:
        coord_r = np.sqrt(coord[S]**2 + coord[Z]**2)
        field_new[R] = (coord[S] * field[S] + coord[Z] * field[Z]) / coord_r
        field_new[H] = (coord[Z] * field[S] - coord[S] * field[Z]) / coord_r
        field_new[PH] = field[PH2]
    return field_new

def get_var_list(dim, var, default=None):
    
    if isinstance(default, Iterable) and not isinstance(default, str):
        if len(default) == 1:
            default = default*dim
        else:
            assert len(default) == dim, "Length of array not equal to number of dimensions."
    else:
        default = [default]*dim
    if isinstance(var, Iterable) and not isinstance(var, str):
        if len(var) == 1:
            var = var*dim
        else:
            assert len(var) == dim, "Length of array not equal to number of dimensions."
    else:
        var = default if var==None else [var]*dim
    return var

class Hist(object):
    '''
    Args
    field: histogram field(s)
    vmin, vmax: minimum and maximum values of the field(s)
    dim: dimension of the histogram
    name, binname: name of histogram and bin array(s)
    weight: histogram weight
    nbin: number of bins in each dimension
    do_log: use the log of the field
    idx_event: use star birth or star death events
    '''
    def __init__(self, field, vmin, vmax, dim=1, name=None, binname=None, weight=None, nbin=None, do_log=None, idx_event=BIRTH, do_trunc=True):
        
        self.dim = dim
        self.field_list = get_var_list(self.dim, field)
        self.vmin_list = get_var_list(self.dim, vmin)
        self.vmax_list = get_var_list(self.dim, vmax)
        self.weight = weight
        self.nbin_list = get_var_list(self.dim, nbin, 512)
        self.do_log_list = get_var_list(self.dim, do_log, True)
        self.idx_event = idx_event
        self.do_trunc = do_trunc

        name_str_prepend = '_'.join(self.field_list) if name == None else name
        name_str_append = '%dd%s' % (self.dim, ['', '_death'][self.idx_event])
        self.name = name_str_prepend
        self.binname_list = get_var_list(self.dim, binname, self.field_list)
        self.binname_list = [binname + name_str_append for binname in self.binname_list]
        self.histname = name_str_prepend + "_hist" + name_str_append
        self.pdfname = name_str_prepend + "_pdf" + name_str_append

        self.bin_width_list, self.bin_center_list = [], []
        for i in range(self.dim):
            if self.do_log_list[i]:
                self.bin_width_list.append((np.log10(self.vmax_list[i]) - np.log10(self.vmin_list[i])) / self.nbin_list[i])
                self.bin_center_list.append(10**(np.linspace(np.log10(self.vmin_list[i]), np.log10(self.vmax_list[i]), self.nbin_list[i]+1) + self.bin_width_list[i]/2)[:-1])
            else:
                self.bin_width_list.append((self.vmax_list[i] - self.vmin_list[i]) / self.nbin_list[i])
                self.bin_center_list.append((np.linspace(self.vmin_list[i], self.vmax_list[i], self.nbin_list[i]+1) + self.bin_width_list[i]/2)[:-1])

    def calc_hist(self, stardata):
        field_list, range_list = [], []
        if np.sum(stardata['event']==self.idx_event) == 0: 
            hist = np.zeros(self.nbin_list)
        else:
            for i in range(self.dim):
                field = stardata[self.field_list[i]][stardata['event']==self.idx_event]
                if self.do_log_list[i] == True:
                    small_number = const.small * np.mean(field)
                    field_list.append(np.log10(field + small_number))
                    range_list.append((np.log10(self.vmin_list[i]), np.log10(self.vmax_list[i])))
                else:
                    field_list.append(field)
                    range_list.append((self.vmin_list[i], self.vmax_list[i]))
            args_histdd = {
                'sample': np.array(field_list).T,
                'bins': self.nbin_list,
                'range': range_list,
                'weights': np.ones(np.sum(stardata['event']==self.idx_event)) if self.weight == None else stardata[self.weight][stardata['event']==self.idx_event]
            }
            hist, _ = np.histogramdd(**args_histdd)
        return hist
    
    def calc_pdf(self, hist):
        return hist / np.sum(hist) / np.prod(self.bin_width_list)

def calc_hist(field, vmin=None, vmax=None, weight=None, nbin=256, do_log=True, do_norm=False):
    '''
    Compute the histogram of a field.

    Args
    field: field
    vmin, vmax (float): min and max field values
    weight: weight
    nbin (int): number of bins
    do_log (bool): use the log of the field
    do_norm (bool): normalize the histogram i.e. generate a PDF

    Returns
    field1d: field bin centers
    hist: histogram of the field
    '''
    if vmin == None: vmin = np.min(field)
    if vmax == None: vmax = np.max(field)
    if np.all(weight) == None: weight = np.ones_like(field)
    if do_log: field, vmin, vmax = np.log10(field), np.log10(vmin), np.log10(vmax)
    hist, bin_edge = np.histogram(field.flatten(), weights=weight.flatten(), bins=nbin, range=(vmin, vmax))
    bin_width = np.diff(bin_edge)[0]
    field1d = bin_edge[:-1] + bin_width/2
    if do_log: field1d = 10**field1d
    if do_norm: hist /= (np.sum(weight) * bin_width)
    return field1d, hist

def calc_profile1d(coord, field, vmin=None, vmax=None, weight=None, nbin=256, do_log=True, do_cum=False):
    '''
    Compute a 1d profile of field with respect to a coordinate.

    Args
    coord: coordinate
    field: field
    vmin, vmax (float): min and max field values
    weight: weight
    nbin (int): n umber of bins
    do_log (bool): use the log of the field
    do_cum (bool): cumulative 1d profile

    Returns
    coord1d: coord bin centers
    field1d: 1d profile of field with respect to coordinate
    '''
    if vmin == None: vmin = np.min(coord)
    if vmax == None: vmax = np.max(coord)
    if np.all(weight) == None: weight = np.ones_like(field)
    coord1d, weighted_field1d = calc_hist(coord, vmin=vmin, vmax=vmax, weight=(field * weight), nbin=nbin, do_log=do_log, do_norm=False)
    coord1d, weight1d = calc_hist(coord, vmin=vmin, vmax=vmax, weight=weight, nbin=nbin, do_log=do_log, do_norm=False)
    if do_cum:
        field1d = np.cumsum(weighted_field1d) / np.cumsum(weight1d)
    else:
        field1d = weighted_field1d / weight1d
    return coord1d, field1d

def calc_hist2d(field1, field2, vmin1=None, vmax1=None, vmin2=None, vmax2=None, weight=None, nbin=30, do_log1=True, do_log2=True, do_norm=False):
    '''
    Compute distribution of a quantity in a two-dimensional phase space.

    Args
    field1, field2: fields
    vmin1, vmax1, vmin2, vmax2 (float): min and max field values
    weight: weight for the bins
    nbin (int): number of bins
    do_log1, do_log2 (bool): use the log of the field
    do_norm (bool): normalize the histogram i.e. generate a pdf

    Returns
    field12d, field22d: field bin centers
    hist2d: 2d histogram
    '''
    if vmin1 == None: vmin1 = np.min(field1)
    if vmax1 == None: vmax1 = np.max(field1)
    if vmin2 == None: vmin2 = np.min(field2)
    if vmax2 == None: vmax2 = np.max(field2)
    if np.all(weight) == None: weight = np.ones_like(field1)
    if do_log1: field1, vmin1, vmax1 = np.log10(field1), np.log10(vmin1), np.log10(vmax1)
    if do_log2: field2, vmin2, vmax2 = np.log10(field2), np.log10(vmin2), np.log10(vmax2)
    hist2d, xbin_edge, ybin_edge = np.histogram2d(field1.flatten(), field2.flatten(), weights=weight.flatten(), bins=(nbin, nbin), range=[(vmin1, vmax1), (vmin2, vmax2)])
    xbin_width, ybin_width = np.diff(xbin_edge)[0], np.diff(ybin_edge)[0]
    field12d = xbin_edge[:-1] + xbin_width/2
    field22d = ybin_edge[:-1] + ybin_width/2
    if do_log1: field12d = 10**field12d
    if do_log2: field22d = 10**field22d
    hist2d = hist2d.T
    if do_norm: hist2d /= (np.sum(weight) * xbin_width * ybin_width)
    return field12d, field22d, hist2d

def plot_pdf2d(field1, field2, vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None, weight=None, nbin=30, do_log1=True, do_log2=True, do_log3=True, fig=None):
    '''
    Compute distribution of a quantity in a two-dimensional phase space.

    Args
    field1, field2: fields
    vmin1, vmax1, vmin2, vmax2, vmin3, vmax3 (float): min and max field values
    weight: weight for the bins
    nbin (int): number of bins
    do_log1, do_log2, do_log3 (bool): use the log of the field
    fig: figure object

    Returns
    axs: figure axes
    field12d, field2d: field bin centers
    pdf2d: 2d PDF
    im: image object
    '''
    if fig == None: fig = plt.figure(figsize=(4, 4))
    
    ax1 = fig.add_axes([0, 0, 1, 1])
    field12d, field22d, pdf2d = calc_hist2d(field1, field2, vmin1=vmin1, vmax1=vmax1, vmin2=vmin2, vmax2=vmax2, weight=weight, nbin=nbin, do_log1=do_log1, do_log2=do_log2, do_norm=True)
    if vmin3 == None: vmin3 == np.min(pdf2d[pdf2d > 0])
    if vmax3 == None: vmax3 == np.max(pdf2d[pdf2d > 0])
    if do_log3: pdf2d, vmin3, vmax3 = np.log10(pdf2d + const.small), np.log10(vmin3), np.log10(vmax3)
    im = ax1.pcolormesh(field12d, field22d, pdf2d, vmin=vmin3, vmax=vmax3)
    if do_log1: ax1.set_xscale('log')
    if do_log2: ax1.set_yscale('log')
    
    ax2 = fig.add_axes([0, 1, 1, 0.3], sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    field11d, pdf11d = calc_hist(field1, vmin=vmin1, vmax=vmax1, weight=weight, nbin=nbin, do_log=do_log1, do_norm=True)
    ax2.plot(field11d, pdf11d, lw=2)
    
    ax3 = fig.add_axes([1, 0, 0.3, 1], sharey=ax1)
    plt.setp(ax3.get_yticklabels(), visible=False)
    field21d, pdf21d = calc_hist(field2, vmin=vmin2, vmax=vmax2, weight=weight, nbin=nbin, do_log=do_log2, do_norm=True)    
    ax3.plot(pdf21d, field21d, lw=2)

    axs = [ax1, ax2, ax3]
    if do_log3: pdf2d = 10**(pdf2d) - const.small

    return axs, field12d, field22d, pdf2d, im


def get_biggest_halo_coord_cubic(aexp):
    ''' Estimate the coordinates of the biggest halo using a cubic fit to each coordinate. '''
    biggest_halo_coord = np.array([cubic(aexp, *halo_poptx), cubic(aexp, *halo_popty), cubic(aexp, *halo_poptz)])
    return biggest_halo_coord

def get_dump(aexp):
    ''' Get the dump number for a given expansion factor. '''
    dump_list = get_dump_list()
    aexp_list = np.array([get_info(dump).aexp for dump in dump_list])
    dump_idx = np.argmin(np.abs(aexp_list - aexp))
    dump = dump_list[dump_idx]
    return dump
    
def get_dump_list():
    ''' Get array of dump numbers. '''
    num_dump = []
    for filename in sorted(os.listdir()):
        if filename[:6] == 'output' and os.path.isdir(filename):
            num_dump.append(int(filename[7:]))
    num_dump = np.array(num_dump)
    return num_dump

def get_info(dump):
    ''' 
    Get information about a given dump. 
    
    Args
    dump (int): dump number
    
    Returns
    info: info SimpleNamespace
        ncpu (int): number of cpus
        amr_level_coarse (int): number of coarse AMR levels
        amr_level_sim_max (int): maximum AMR level at any expansion factor
        amr_level_reduce_exp (int): number of AMR levels locked by expansion factor
        amr_level_max (int): maximum AMR level at current expansion factor
        aexp (float): expansion factor
        H0 (float): Hubble constant at z=0
        Omega_m0 (float): mass density parameter at z=0
        Omega_L0 (float): dark energy density parameter at z=0
        Omega_k0 (float): curvature density parameter at z=0
        Omega_b0 (float): baryon density parameter at z=0
        length_unit (float): cm per code unit of length
        density_unit (float): g/cm^3 per code unit of density
        time_unit (float): s per code unit of time
        mass_unit (float): g per code unit of mass
        velocity_unit (float): cm/s per code unit of velocity
        energy_unit (float): erg per code unit of energy
        energy_density_unit (float): erg/cm^3 per code unit of energy density
    '''
    filename = "output_%.5d/info_%.5d.txt" % (dump, dump)
    info = {}
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()[:18]):
            if i == 0: info['ncpu'] = int(line.split()[2])
            if i == 2: info['amr_level_coarse'] = int(line.split()[2])
            if i == 3: info['amr_level_sim_max'] = int(line.split()[2])
            if i == 9: info['aexp'] = float(line.split()[2])
            if i == 10: info['H0'] = float(line.split()[2])
            if i == 11: info['Omega_m0'] = float(line.split()[2])
            if i == 12: info['Omega_L0'] = float(line.split()[2])
            if i == 13: info['Omega_k0'] = float(line.split()[2])
            if i == 14: info['Omega_b0'] = float(line.split()[2])
            if i == 15: info['length_unit'] = float(line.split()[2])
            if i == 16: info['density_unit'] = float(line.split()[2])
            if i == 17: info['time_unit'] = float(line.split()[2])
    info = SimpleNamespace(**info)
    info.mass_unit = info.density_unit * info.length_unit**3
    info.vel_unit = info.length_unit / info.time_unit
    info.energy_unit = info.mass_unit * info.vel_unit**2
    info.energy_density_unit = info.density_unit * info.vel_unit**2
    info.amr_level_reduce_exp = -min(-4, int(np.floor(np.log2(info.aexp))))
    info.amr_level_max = info.amr_level_sim_max - info.amr_level_coarse - info.amr_level_reduce_exp
    info.H0 *= const.km / const.Mpc
    return info

def move_to_sim_dir(sim_round, sim_name, do_print=True):
    ''' Move to simulation directory. '''
    sim_dir = os.path.join(sim_base_dir, "round%d" % sim_round, sim_name)
    os.chdir(sim_dir)
    if do_print: print("Moving to directory '%s'." % sim_dir)
    return sim_dir

def add_cbar_to_ax(im, ax=None, ticks=None, label=None, size="5%", pad=0.1, orientation='vertical', extend='neither'):
    '''
    Add a colorbar to the axis.

    Args
    im: image object
    ax: axis object
    ticks: array of ticks
    label (str): label
    orientation (str): colorbar orientation
    size (str): colorbar size
    pad (float): colorbar padding

    Returns 
    cbar: colorbar object
    '''
    if ax == None: ax = plt.gca()
    divider = make_axes_locatable(ax)
    if orientation == "horizontal":
        cax = divider.append_axes("top", size=size, pad=pad)
        cbar = plt.colorbar(im, cax=cax, ticks=ticks, label=label, orientation="horizontal", extend=extend)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    elif orientation == "vertical":
        cax = divider.append_axes("right", size=size, pad=pad)
        cbar = plt.colorbar(im, cax=cax, ticks=ticks, label=label, orientation="vertical", extend=extend)
    return cbar

def add_cbar_to_fig(im, fig, ticks=None, label=None, bbox=[.9, .11, .02, .77], extend='neither'):
    ''' Add a colorbar to the figure. '''
    cax = fig.add_axes(bbox)
    cbar = plt.colorbar(im, cax=cax, ticks=ticks, label=label, extend=extend)
    return cbar

def add_custleg_to_ax(label_list, color_list, linestyle_list=None, loc=None, lw=2, ax=None, title=None):
    ''' Add a custom legend to the axis. '''
    if ax == None: ax = plt.gca()
    if linestyle_list == None: linestyle_list = ['-'] * len(label_list)
    custom_line_list = [Line2D([0], [0], color=color, lw=lw, linestyle=linestyle) for color, linestyle in zip(color_list, linestyle_list)]
    ax.legend(custom_line_list, label_list, loc=loc)

def get_numline(filename):
    ''' Quickly compute the number of lines in a file. '''
    with open(filename, "rbU") as f:
        num_line = sum(1 for _ in f)
    return num_line

def downsample_hist(bin_center, hist, fac_ds):
    ''' Downsample a histogram. '''
    bin_center_ds = bin_center[::fac_ds]
    if hist.size % fac_ds != 0: hist = hist[:-(hist.size%fac_ds)]
    hist_ds = np.sum(hist.reshape(fac_ds, hist.size//fac_ds), axis=0) / fac_ds
    return bin_center_ds, hist_ds
