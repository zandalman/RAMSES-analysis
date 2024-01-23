from modules import *

linear = lambda x, a: a*x
affine = lambda x, a, b: a + b*x
quadratic = lambda x, a, b, c: a + b*x + c*x**2
cubic = lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3
powerlaw = lambda x, a, b: a*x**b
class Terminal:
    ''' Context manager for running commands in the terminal from Python. '''
    def __enter__(self):
        self.cwd = os.getcwd()
        return None
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self.cwd)

def median_weighted(field, weights):
    ''' Get the weighted median of a distribution. '''
    idx_sorted = np.argsort(field)
    cumsum = np.cumsum(weights[idx_sorted])
    field_median = field[idx_sorted[np.searchsorted(cumsum, 0.5 * cumsum[-1])]]
    return field_median

def plot_pdf(field, extrema, ax=None, weights=None, nbins=200, do_log=True, label='misc var', show_median=False, show_mean=False, do_axes_labels=True, flip_axes=False):
    ''' 
    Plot a PDF of a distribution. 
    
    Args
    ax: matplotlib axis object
    field: field
    extrema: tuple of min and max field value
    weights: weights
    nbins (int): number of bins
    do_log (bool): use the log of the field
    label (str): label for the PDF
    show_median (bool): plot a vertical line for the median of the distribution
    show_mean (bool): plot a vertical line for the mean of the distribution
    flip_axes (bool): flip the x and y axes
    '''
    if ax == None: ax = plt.gca()
    if np.all(weights) == None: weights = np.ones_like(field)
    if do_log: 
        field_new = np.log10(field + epsilon)
    else:
        field_new = field
    if do_log: extrema = (np.log10(extrema[0]), np.log10(extrema[1]))
    hist, bins = np.histogram(field_new, weights=weights, bins=nbins, range=extrema)
    field_bins = bins[:-1] + np.diff(bins)[0]
    if do_log: field_bins = 10**field_bins
    pdf = hist
    pdf = pdf / np.sum(weights) / np.diff(bins)[0] # normalize the pdf
    if show_median: field_median = median_weighted(field, weights)
    if show_mean: field_mean = np.sum(field * weights) / np.sum(weights)
    if flip_axes:
        ax.plot(pdf, field_bins, lw=2)
        if do_log: ax.set_yscale('log')
        if do_axes_labels:
            ax.set_ylabel(label)
            ax.set_xlabel('PDF')
        if show_median: ax.axhline(y=field_median, lw=2, color='red', label=('median ' + label + r'$ = %.3g$' % field_median))
        if show_mean: ax.axhline(y=field_mean, lw=2, color='orange', label=('mean ' + label + r'$ = %.3g$' % field_mean))
    else:
        ax.plot(field_bins, pdf, lw=2)
        if do_log: ax.set_xscale('log')
        if do_axes_labels:
            ax.set_xlabel(label)
            ax.set_ylabel('PDF')
        if show_median: ax.axvline(x=field_median, lw=2, color='red', label=('median ' + label + r'$ = %.3g$' % field_median))
        if show_mean: ax.axvline(x=field_mean, lw=2, color='orange', label=('mean ' + label + r'$ = %.3g$' % field_mean))

def get_stdout(cmd):
    ''' Return the standard output of a command line directive '''
    stdout = subprocess.check_output(cmd, shell=True).decode()
    return stdout

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
    
def calc_eps_sf(density, energy_turb, temp, dx, b_turb=1.0, gamma=5/3):
    ''' 
    Star formation efficiency in the multi-freefall model.
    See Federrath&Klessen2012 (https://arxiv.org/pdf/1209.2856.pdf) and Kretschmer&Teyssier2021 (https://arxiv.org/pdf/1906.11836.pdf) for details.

    Args
    rho: density
    energy_turb: turbulent energy
    temp: temperature
    dx: resolution
    b_turb: turbulence forcing parameter
        varies smoothly between b ~ 1/3 for purely solenoidal (divergence-free) forcing 
        and b ~ 1 for purely compressive (curl-free) forcing
        A stochastic mixture of forcing modes in 3-d space leads to b ~ 0.4
    gamma: adiabatic index

    Returns
    epsilon_SF: star formation efficiency
    '''
    c_s = np.sqrt(gamma * const.k_B * temp / const.m_p) # sound speed
    mach_turb = np.sqrt(2/3 * energy_turb) / c_s # turbulent Mach number
    alpha_vir = 15 / np.pi * c_s**2 * (1 + mach_turb**2) / (const.G * density * dx**2) # virial parameter
    s_crit = np.log(alpha_vir * (1 + 2 * mach_turb**4 / (1 + mach_turb**2))) # lognormal critical density for star formation
    sigma_s = np.sqrt(np.log(1 + b_turb**2 * mach_turb**2)) # standard deviation of the lognormal subgrid density distribution
    eps_sf = 1/2 * np.exp(3/8 * sigma_s**2) * (1 + erf((sigma_s**2 - s_crit) / np.sqrt(2 * sigma_s**2))) # star formation efficiency
    return eps_sf

def a_exp_to_proper_time(a, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert expansion factor to proper time.'''
    integrand = lambda a: (Omega_m0 * a**(-1) + Omega_k0 + Omega_L0 * a**2)**(-1/2)
    t = quad(integrand, 0, a)[0] / H0
    return t

def a_exp_to_conformal_time(a, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert expansion factor to conformal time.'''
    integrand = lambda a: (Omega_m0 * a + Omega_k0 * a**2 + Omega_L0 * a**4)**(-1/2)
    tau = const.c * quad(integrand, 0, a)[0] / H0
    return tau

def proper_time_to_a_exp(t, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert proper time to expansion rate.'''
    a = fsolve(lambda a: (a_exp_to_proper_time(a, Omega_m0, Omega_k0, Omega_L0, H0) - t) * H0, 0.1)
    return a

def conformal_time_to_a_exp(tau, Omega_m0=const.Omega_m0, Omega_k0=const.Omega_k0, Omega_L0=const.Omega_L0, H0=const.H0):
    ''' Convert conformal time to expansion rate.'''
    a = fsolve(lambda a: (a_exp_to_conformal_time(a, Omega_m0, Omega_k0, Omega_L0, H0) - tau) * H0, 0.1)
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

def calc_1d_profile(x, y, extrema=None, do_log_x=True, nbins=100, weight=None, cond=None, do_cum=False):
    '''
    Compute radial profile of field averaged over spherical shells.

    Args
    x, y: quantities with which to 
    extrema: tuple of max and min field values
    do_log_x (bool): space bins logorithmically
    nbins (int): number of bins
    weight: weight array
    cond: conditional array to select a specific region
    do_cum (bool): cumulative 1d profile

    Returns
    r_1d: array of radial bins
    field_1d: array of spherically averaged field values
    '''
    if do_log_x:
        x = np.log10(x)
        extrema = (np.log10(extrema[0]), np.log10(extrema[1]))
    
    if np.all(cond) == None: cond = 1.
    if np.all(weight) == None: weight = 1.

    counts1, bins = np.histogram(x.flatten(), weights=(y * weight * cond).flatten(), bins=nbins, range=extrema)
    counts2, bins = np.histogram(x.flatten(), weights=(np.ones_like(y) * weight * cond).flatten(), bins=nbins, range=extrema)

    x_1d = bins[:-1] + np.diff(bins)[0]
    if do_cum:
        y_1d = np.cumsum(counts1) / np.cumsum(counts2)
    else:
        y_1d = counts1 / counts2

    if do_log_x: x_1d = 10**x_1d

    return x_1d, y_1d

def calc_phase(field1, field2, extrema1, extrema2, do_log1=True, do_log2=True, nbins=30, cond=None, weight=None):
    '''
    Compute distribution of a quantity in a two-dimensional phase space.

    Args
    field1, field2: fields
    extrema1, extrema2: tuples of max and min field values
    do_log1, do_log2 (bool): space bins logorithmically
    nbins (int): number of bins
    cond: conditional array to select a specific region
    weight: weight for the bins

    Returns
    field1_2d, field2_2d: array of field bins
    weight_2d: array of weight per bin
    '''
    if do_log1: 
        field1 = np.log10(field1)
        extrema1 = (np.log10(extrema1[0]), np.log10(extrema1[1]))
    if do_log2:
        field2 = np.log10(field2)
        extrema2 = (np.log10(extrema2[0]), np.log10(extrema2[1]))
    if np.all(weight) == None: weight = 1.
    if np.all(cond) == None: cond = 1.
    hist, x_bins, y_bins = np.histogram2d(field1.flatten(), field2.flatten(), weights=(weight * cond).flatten(), bins=(nbins, nbins), range=[extrema1, extrema2])
    field1_2d = x_bins[:-1] + np.diff(x_bins)[0]
    field2_2d = y_bins[:-1] + np.diff(y_bins)[0]
    if do_log1: field1_2d = 10**field1_2d
    if do_log2: field2_2d = 10**field2_2d
    weight_2d = hist.T
    return field1_2d, field2_2d, weight_2d

def get_biggest_halo_coord_cubic(a_exp):
    '''
    Estimate the coordinates of the biggest halo using a cubic fit to each coordinate.
    '''
    biggest_halo_coord = np.array([cubic(a_exp, *halo_poptx), cubic(a_exp, *halo_popty), cubic(a_exp, *halo_poptz)])
    return biggest_halo_coord

def get_dump(a_exp):
    ''' Get the dump number for a given expansion factor. '''
    list_of_dump = get_list_of_dump()
    list_of_a_exp = np.array([get_info(dump).a_exp for dump in list_of_dump])
    dump_idx = np.argmin(np.abs(list_of_a_exp - a_exp))
    dump = list_of_dump[dump_idx]
    return dump
    
def get_list_of_dump():
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
        amr_level_couarse (int): number of coarse AMR levels
        amr_level_sim_max (int): maximum AMR level at any expansion factor
        amr_level_reduce_exp (int): number of AMR levels locked by expansion factor
        amr_level_max (int): maximum AMR level at current expansion factor
        a_exp (float): expansion factor
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
            if i == 9: info['a_exp'] = float(line.split()[2])
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
    info.amr_level_reduce_exp = -min(-4, int(np.floor(np.log2(info.a_exp))))
    info.amr_level_max = info.amr_level_sim_max - info.amr_level_coarse - info.amr_level_reduce_exp
    info.H0 *= const.km / const.Mpc
    return info

def move_to_sim_dir(sim_round, sim_name, do_print=True):
    ''' Move to simulation directory. '''
    sim_dir = os.path.join(sim_base_dir, "round%d" % sim_round, sim_name)
    os.chdir(sim_dir)
    if do_print: print("Moving to directory '%s'." % sim_dir)
    return sim_dir
