from modules import *
import sim_config

X, Y, Z = 0, 1, 2 # cartesian coordinate indices
S, PH2, Z = 0, 1, 2 # cylindrical coordinate indices
R, H, PH = 0, 1, 2 # spherical coordinate indices
CART, CYL, SPH = 0, 1, 2 # coordinate systems
HYDRO, DM, STAR = 0, 1, 2
epsilon = 1e-30 # small number

analysis_dir = "/home/za9132/analysis"
save_dir = os.path.join(analysis_dir, "figures", "current")
sim_base_dir = "/home/za9132/scratch/romain"

sim_name_to_latex = {
    "alpha_eps0p01": r"$\varepsilon_{\rm SF} = 0.01$", 
    "alpha_eps0p1": r"$\varepsilon_{\rm SF} = 0.1$", 
    "alpha_eps1p0": r"$\varepsilon_{\rm SF} = 1.0$", 
    "alpha_eps0p01_highres": r"$\varepsilon_{\rm SF} = 0.01$ (highres)", 
    "alpha_eps0p1_highres": r"$\varepsilon_{\rm SF} = 0.1$ (highres)", 
    "alpha_eps1p0_highres": r"$\varepsilon_{\rm SF} = 1.0$ (highres)",
    "dmo": "Dark Matter Only", 
    "gas": "Multi-Freefall Model"
}

# define custom colormaps
varying_color = np.array([np.linspace(0, 1, 100)] * 3).T
const_color = np.array([[0, 0, 0], [1, 0, 0]])
cmap_red = dict(red=varying_color, green=const_color, blue=const_color)
cmap_green = dict(red=const_color, green=varying_color, blue=const_color)
cmap_blue = dict(red=const_color, green=const_color, blue=varying_color)
rgb_red = LinearSegmentedColormap('rgb_red', segmentdata=cmap_red)
rgb_green = LinearSegmentedColormap('rgb_green', segmentdata=cmap_green)
rgb_blue = LinearSegmentedColormap('rgb_blue', segmentdata=cmap_blue)

class Terminal:
    ''' Context manager for running commands in the terminal from Python. '''
    def __enter__(self):
        self.cwd = os.getcwd()
        return None
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self.cwd)


class Arglist(object):
    ''' Class for lists of function arguments. '''
    def __init__(self, args):
        self.args = np.array(args)
        
        
def git_commit(git_message=None):
    '''
    Commit relevant files to git and push changes.
    
    Args
    git_message (str): message for the commit
    '''
    with Terminal() as terminal:
        
        os.chdir(analysis_dir)
        if git_message == None: git_message = "pushing updates to analysis code"
        list_of_filename = ["analysis.ipynb", "yt_to_numpy.ipynb", "analytic.ipynb", "const.py", "sim.py", "modules.py", "read_ramses.py"]
        
        for filename in list_of_filename:
            os.system("git add %s" % filename)
        os.system("git commit -m '%s'" % git_message)
        os.system("git push")
        
        
def clear_figures():
    '''
    Move all current figures to legacy folder.
    '''
    with Terminal() as terminal:
        
        os.chdir(save_dir)
        for dir in os.listdir():
            if os.path.isdir(dir) and dir != "legacy":
                for subdir in os.listdir(os.path.join(".", dir)):
                    subdir_path = os.path.join(".", dir, subdir)
                    subdir_path_legacy = os.path.join(".", "legacy", dir, subdir)
                    if not os.path.isdir(subdir_path): os.mkdir(subdir_path)
                    if len(os.listdir(subdir_path)) > 0:
                        os.system("mv %s/* %s/" % (subdir_path, subdir_path_legacy))


def save_fig(fig_name, filetype="png", dpi=300, round=None, subdir="all"):
    '''
    Save the current matplotlib figure.

    Args
    name (string): figure name
    filetype (string): file type
    dpi (int): dots per inch
    round (int): round of simulation runs
    subdir (str): subdirectory
    '''
    datetime_string = datetime.now().strftime("%m%d%Y%H%M")
    filename = "%s-%s.%s" % (fig_name, datetime_string, filetype)
    if round == None:
        my_save_dir = os.path.join(save_dir, subdir)
    else: 
        if not os.path.isdir(os.path.join(save_dir, "round%d")): os.mkdir(os.path.join(save_dir, "round%d"))
        my_save_dir = os.path.join(save_dir, "round%d", subdir)
    if not os.path.isdir(my_save_dir): os.mkdir(my_save_dir)
    plt.savefig(os.path.join(my_save_dir, filename), bbox_inches="tight", dpi=dpi)
    print("Saved figure as '%s'" % filename)
        

def norm(A):
    ''' Compute the norm of a vector field. '''
    return np.sqrt(np.sum(A**2, axis=0))


def dot(A, B):
    ''' Compute the dot product of two vector fields. '''
    return np.sum(A * B, axis=0)


def proj(A, B, do_norm=True):
    ''' Compute the projection of one vector field onto another. '''
    if do_norm:
        return dot(A, B) / norm(B)
    else:
        return (dot(A, B) / norm(B)**2)[None, :, :, :] * B
    

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
        
    
class Sim(object):
    '''
    Simulation object.
    
    Args
    sim_round (int): round of simulation runs
    sim_name (str): name of the simulation
    npz_file (str): filename of the npz file with the simulation data
    
    Attrs
    sim_idx (int): index of the simulation (ALPHA_EPS0P01, ALPHA_EPS0P1, ALPHA_EPS1P0, DMO, GAS)
    npz_file (str): filename of the npz file with the simulation data
    
    cond_hydro: conditional array to select a specific region for hydro fields
    cond_dm: conditional array to select a specific region for dark matter fields
    cond_star: conditional array to select a specific region for star fields

    halo_idx (int): index of the halo
    halo_mass (float): mass of the halo
    a_exp (float): expansion factor
    redshift (float): redshift
    time_now (float): proper time
    universe_age (float): age of the Universe (i.e. proper time at a_exp = 1)
    
    H0 (float): Hubble constant at z=0
    Omega_m0 (float): mass density parameter at z=0
    Omega_L0 (float): dark matter density parameter at z=0
    Omega_k0 (float): curvature density parameter at z=0
    Omega_b0 (float): baryon density parameter at z=0
    H (float): Hubble constant
    rho_crit (float): critical density
    
    length_unit (float): cm per code unit of length
    density_unit (float): g/cm^3 per code unit of density
    time_unit (float): s per code unit of time
    mass_unit (float): g per code unit of mass
    velocity_unit (float): cm/s per code unit of velocity
    energy_unit (float): erg per code unit of energy
    energy_density_unit (float): erg/cm^3 per code unit of energy density
    
    ref_crit: refinement criterion
    contamination_frac: contamination fraction of high mass dark matter particles (cached property)
    max_amr_level (int): max AMR level of data read from yt
    lowres (int): factor by which grid resolution is lower than resolution of max AMR level
    box_size (float): simulation box size
    left_edge: array of coordinate of the left corner of the simulation box
    N (int): array size
    
    coord: coordinate array
    coord1d: 1d coordinate array
    dx (float): length element
    dA (float): area element
    dV (float): volume element
    coord_cyl_at_cart: cylindrical coordinates on the Cartesian grid (cached property)
    coord_sph_at_cart: spherical coordinate on the Cartesian grid (cached property)
    
    coord_sph: spherical coordinate array
    coord1d_sph: 1d spherical coordinate array
    dx_sph: spherical coordinate length elements
    dA_hph: theta-phi area element
    coord_AH_at_sph: Aitoff-Hammer projection coordinates on the spherical grid
    coord_cart_at_sph: Cartesian coordinates on the spherical grid (cached property)
    coord_cyl_at_sph: cylindrical coordinates on the spherical grid (cached property)
    
    coord_dm: coordinates of dark matter particles
    coord_sph_dm: spherical coordinates of dark matter particles (cached property)
    mass_dm: dark matter particle masses
    
    coord_star: coordinates of star particles
    coord_sph_star: spherical coordinates of star particles
    mass_star: star particle masses
    time_starbirth: star particle birth times
    age_star: star particle ages
    
    density: density
    n_H: Hydrogen number density (cached property)
    density_dust: dust density (cached property)
    density_star: density of stellar matter (cached property)
    density_dm: density of dark matter (cached property)
    metallicity: metallicity
    pressure: pressure
    energy_turb: turbulent energy
    temp: temp (cached property)
    c_s: sound speed (cached property)
    mach: Mach number (cached property)
    mach_turb: turbulent Mach number (cached property)
    alpha_vir: virial parameter (cached property)
    ion_frac: ionization fraction (cached property)
    SFR_density: star formation rate density (cached property)
    
    vel_vec: velocity vector
    vel_vec_sph_at_cart: velocity vector in spherical coordinates on the Cartesian grid (cached property)
    vel: magnitude of the velocity (cached property)
    vel_turb: turbulent velocity (cached property)
    ang_mom: angular momentum (cached property)
    
    sim_latex (str): latex string of the simulation, for plotting
    sim_dir (str): path to the simulation data
    save_dir (str): path of the directory to save figures
    
    gamma (float): adiabatic index
    epsilon_SF: array of star formation efficiencies
    summury_stats: list of Stat objects
    '''
    def __init__(self, sim_round, sim_name, npz_file, epsilon_SF=None):
        
        self.sim_round = sim_round
        self.sim_name = sim_name
        self.sim_latex = sim_name_to_latex[self.sim_name]
        self.sim_dir = os.path.join(sim_base_dir, "round%d" % self.sim_round, self.sim_name)
        self.save_dir = os.path.join(save_dir, "round%d" % self.sim_round, self.sim_name)
        
        os.chdir(self.sim_dir)
        print("Moving to directory '%s'." % self.sim_dir)
        
        self.epsilon_SF = epsilon_SF
        self.alpha_vir_crit = 10.
        self.gamma = 5/3
        
        self.npz_file = npz_file
        if self.npz_file == None:
            print("Pick an available npz file:")
            for file in os.listdir():
                if file[-4:] == ".npz": print(file)
            return
        else:
            assert self.npz_file in os.listdir(), "File '%s' not in simulation directory." % self.npz_file
            data = np.load(self.npz_file)
            for var_name in data:
                setattr(self, var_name, data[var_name])
        
        self.coord1d = np.array([
            self.coord[X, :, self.N//2, self.N//2], 
            self.coord[Y, self.N//2, :, self.N//2], 
            self.coord[Z, self.N//2, self.N//2, :]
        ])
        self.dx = np.diff(self.coord1d[X])[0]
        self.dA = self.dx**2
        self.dV = self.dx**3
            
        self.redshift = 1 / self.a_exp - 1
        self.H = self.H0 * np.sqrt(self.Omega_m0 / self.a_exp**3 + self.Omega_k0 / self.a_exp**2 + self.Omega_L0)
        self.rho_crit = 3 * self.H**2 / (8 * np.pi * const.G)
        self.time_now = self.a_exp_to_proper_time(self.a_exp)
        self.universe_age = self.a_exp_to_proper_time(1.)
        
        self.time_starbirth = (self.tau_starbirth / self.H0 + self.universe_age)
        self.age_star = self.time_now - self.time_starbirth
        
        self.cond_hydro = np.ones_like(self.density, dtype=bool)
        self.cond_dm = np.ones_like(self.mass_dm, dtype=bool)
        self.cond_star = np.ones_like(self.mass_star, dtype=bool)

        self.create_sph_grid()
    
    def save_fig(self, fig_name, filetype="png", dpi=300):
        '''
        Save the current matplotlib figure.

        Args
        name (string): figure name
        filetype (string): file type
        dpi (int): dots per inch
        '''
        save_fig(fig_name, filetype=filetype, dpi=dpi, round=self.sim_round, subdir=self.sim_name)
        
    def create_sph_grid(self):
        '''
        Create a grid in spherical coordinates.
        '''
        self.coord1d_sph = np.array([
            np.linspace(0, self.box_size / 2, self.N),
            np.linspace(0, np.pi, self.N),
            np.linspace(0, 2 * np.pi, self.N)
        ])
        self.coord_sph = np.array(np.meshgrid(self.coord1d_sph[R], self.coord1d_sph[H], self.coord1d_sph[PH], indexing='ij'))
        self.dx_sph = np.diff(self.coord1d_sph, axis=-1)[:, 0]
        self.coord_AH_at_sph = np.array(calc_AH_coords(self.coord_sph[H, self.N//2], self.coord_sph[PH, self.N//2]))
        self.dA_hph = self.coord_sph[R]**2 * np.sin(self.coord_sph[H]) * self.dx_sph[H] * self.dx_sph[PH]
    
    def a_exp_to_proper_time(self, a):
        ''' Convert expansion factor to proper time.'''
        integrand = lambda a: (self.Omega_m0 * a**(-1) + self.Omega_k0 + self.Omega_L0 * a**2)**(-1/2)
        t = quad(integrand, 0, a)[0] / self.H0
        return t

    def a_exp_to_conformal_time(self, a):
        ''' Convert expansion factor to conformal time.'''
        integrand = lambda a: (self.Omega_m0 * a + self.Omega_k0 * a**2 + self.Omega_L0 * a**4)**(-1/2)
        tau = const.c * quad(integrand, 0, a)[0] / self.H0
        return tau

    def proper_time_to_a_exp(self, t):
        ''' Convert proper time to expansion rate.'''
        a = fsolve(lambda a: (self.a_exp_to_proper_time(a) - t) * self.H0, self.a_exp)
        return a

    def conformal_time_to_a_exp(self, tau):
        ''' Convert conformal time to expansion rate.'''
        a = fsolve(lambda a: (self.a_exp_to_conformal_time(a) - tau) * self.H0, self.a_exp)
        return a

    def interp_to_sph(self, field):
        ''' Interpolate a field to a spherical grid '''
        field_sph = interpn(self.coord1d, field, np.moveaxis(self.coord_cart_at_sph, 0, -1), bounds_error=False, fill_value=None)
        return field_sph
    
    @staticmethod
    def get_current_args(args, kwargs, i, j):

        current_args = list(args).copy()
        current_kwargs = kwargs.copy()
        
        for arg_idx, arg in enumerate(args):
            if type(arg).__name__ == 'Arglist':
                current_args[arg_idx] = arg.args[i, j]

        for kwarg_name, kwarg_value in kwargs.items():
            if type(kwarg_value).__name__ == 'Arglist':
                current_kwargs[kwarg_name] = kwarg_value.args[i, j]

        return current_args, current_kwargs
    
    def plot_grid(self, *args, figsize=None, nrows=1, ncols=1, sharex=False, sharey=False, hspace=None, wspace=None, share_cbar=False, plot_type="slice", **kwargs):
        '''
        Wrapper function for plot_slice_on_ax, for a single plot.

        Args
        *args: arguments to be passed to plot_slice_on_ax
        **kwargs: keyword arguments to be passed to plot_slice_on_ax
        figsize: figure size
        sharex (bool): share x-axis
        sharey (bool): share y-axis
        hspace (float): height spacing between subplots
        wspace (float): width spacing between subplots
        plot_type (str): plot type ['slice', 'rgb_slice', 'AH']
        '''
        kwargs = dict(sim_config.defaults, **kwargs)
        kwargs_obj = SimpleNamespace(**kwargs)
        
        _, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=False)
        
        for arg in args:
            if type(arg) == Arglist:
                assert arg.args.shape[:2] == (nrows, ncols), "Arglist shape must match (nrows, ncols)"
        
        for kwarg_value in kwargs.values():
            if type(kwarg_value) == Arglist:
                assert kwarg_value.args.shape[:2] == (nrows, ncols), "Arglist shape must match (nrows, ncols)"
        
        for i, axs1d in enumerate(axs):
            for j, ax in enumerate(axs1d):
                
                do_xlabel = (i == len(axs1d)-1) if sharex else True
                do_ylabel = (j == 0) if sharey else True

                current_args, current_kwargs = self.get_current_args(args, kwargs, i, j)
                if plot_type == "slice":
                    im = self.plot_slice_on_ax(ax, *current_args, **current_kwargs, do_xlabel=do_xlabel, do_ylabel=do_ylabel, do_cbar=(not share_cbar))
                elif plot_type == "AH":
                    im = self.plot_AH_on_ax(ax, *current_args, **current_kwargs, do_xlabel=do_xlabel, do_ylabel=do_ylabel, do_cbar=(not share_cbar))

        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        if share_cbar: plt.colorbar(im, ax=axs, ticks=np.arange(kwargs_obj.extrema[0], kwargs_obj.extrema[1] + 0.5 * kwargs_obj.extrema.cbar_tick_increment, kwargs_obj.extrema.cbar_tick_increment), label=kwargs_obj.extrema.cbar_label)
    
    def plot(self, *args, figsize=None, plot_type="slice", **kwargs):
        '''
        Wrapper function for plot_slice_on_ax, for a single plot.

        Args
        *args: arguments to be passed to plotting function
        **kwargs: keyword arguments to be passed to plotting function
        figsize: figure size
        plot_type (str): plot type ['slice', 'rgb_slice', 'AH']
        '''
        kwargs = dict(sim_config.defaults, **kwargs)
        
        plt.figure(figsize=figsize)
        ax = plt.gca()
        if plot_type == "slice":
            self.plot_slice_on_ax(ax, *args, **kwargs, do_xlabel=True, do_ylabel=True, do_cbar=True)
        elif plot_type == "AH":
            self.plot_AH_on_ax(ax, *args, **kwargs, do_xlabel=True, do_ylabel=True, do_cbar=True)
    
    def get_field_input(self, field, field_name='Field'):
        ''' Return a field given an input, which can either be a field or a string. '''
        if type(field) in [str, np.str_]:
            assert hasattr(self, field), "%s must be an attribute of Sim object if passed as string." % field_name
            field = getattr(self, field)
        return field
    
    def get_field2d(self, field, **kwargs):
        '''
        Get the cross section of a field perpendicular to a coordinate axis.

        Args
        field: field
        kwargs: keyword arguments
            unit (float): unit of the field
            slice (int): index of the coordinate direction perpendicular to the slice plane
            project (bool): project the field along the slice direction
            do_integrate (bool): integrate the physical length of the projection
            weight: weight array to use for the project
            cond: conditional array to select a specific region for the projection
            avg (bool): take the average (rather than sum) in the projection
            do_log (bool): plot the log of the field
            slice_coord (float): coordinate of the cross section
            max_pixels (int): maximum number of pixels to plot in each dimension

        Returns
        coord1_idx, coord2_idx: indices of the coordinates of the 2d field
        coord12d, coord22d: coordinates of the 2d field
        field2d: 2d field
        '''
        kwargs_obj = SimpleNamespace(**kwargs)
        
        field = self.get_field_input(field, "Field")
        weight = self.get_field_input(kwargs_obj.weight, "Weight")
        cond = self.get_field_input(kwargs_obj.cond, "Conditional array")
        
        slice_coord_idx = np.argmin(np.abs(self.coord1d[kwargs_obj.slice] - kwargs_obj.slice_coord))

        coord1_idx, coord2_idx = np.sort([(kwargs_obj.slice + 1) % 3, (kwargs_obj.slice + 2) % 3])
        coord12d = self.coord[coord1_idx].take(indices=slice_coord_idx, axis=kwargs_obj.slice)
        coord22d = self.coord[coord2_idx].take(indices=slice_coord_idx, axis=kwargs_obj.slice)
        
        if kwargs_obj.project:
            if kwargs_obj.do_integrate: weight *= self.dx
            field2d = np.sum(field * weight * cond, axis=kwargs_obj.slice)
            if kwargs_obj.avg: field2d /= np.sum(np.ones_like(field) * weight * cond + epsilon, axis=kwargs_obj.slice)
        else:
            field2d = field.take(indices=slice_coord_idx, axis=kwargs_obj.slice)

        if kwargs_obj.do_log: field2d = np.log10(field2d + epsilon)

        if kwargs_obj.max_pixels != None:
            skip = np.max([1, self.N // kwargs_obj.max_pixels])
            field2d = field2d[::skip, ::skip]
            coord12d = coord12d[::skip, ::skip]
            coord22d = coord22d[::skip, ::skip]

        return coord1_idx, coord2_idx, coord12d, coord22d, field2d

    def plot_slice_on_ax(self, ax, field, extrema, **kwargs):
        '''
        Plot the cross section of a field perpendicular to a coordinate axis.

        Args
        ax: axis object
        field: field
        extrema (tuple): tuple of min and max field value
        kwargs: keyword arguments
            width (float): width of plot
            nlevels (int): number of levels in the contour plot
            cmap (str): colormap for the plot
            cbar_label (string): colorbar label
            cbar_tick_increment (float): increment of the colorbar ticks
            plot_star (bool): plot markers for star particles within one cell of the cross section
            plot_dm (bool): plot markers for dark matter particles within one cell of the cross section
            isocontours: list of field values to plot isocontours
            isocontour_field: field to use for isocontours
            color_isocontour (str): color of isocontours
            color_star (str): color of star particles, only used if plot_star == True
            color_dm (str): color of dark matter particles, only used if plot_dm == True
            do_xlabel (bool): label x-axis
            do_ylabel (bool): label y-axis
            do_cbar (bool): plot a colorbar
            cbar_orientation (str): orientation of the colorbar
            title (str): title of the plot
            keyword arguments of self.get_field2d

        Returns
        im: QuadContourSet object
        '''
        kwargs_obj = SimpleNamespace(**kwargs)

        coord1_idx, coord2_idx, coord12d, coord22d, field2d = self.get_field2d(field, **kwargs)

        if kwargs_obj.cbar_tick_increment == None: kwargs_obj.cbar_tick_increment = (extrema[1] - extrema[0]) / 10
        
        im = ax.contourf(coord12d / const.kpc, coord22d / const.kpc, field2d, extend='both', cmap=kwargs_obj.cmap, levels=np.linspace(extrema[0], extrema[1], kwargs_obj.nlevels))
        ax.set_aspect(True)
        if kwargs_obj.do_cbar: self.plot_cbar(ax, im, extrema, kwargs_obj.cbar_tick_increment, kwargs_obj.cbar_label, kwargs_obj.cbar_orientation)

        coord_labels = [r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]"]
        coord1_label, coord2_label = coord_labels[coord1_idx], coord_labels[coord2_idx]
        if kwargs_obj.do_xlabel: ax.set_xlabel(coord1_label)
        if kwargs_obj.do_ylabel: ax.set_ylabel(coord2_label)

        if kwargs.width != None:
            ax.set_xlim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)
            ax.set_ylim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)

        if kwargs_obj.title != None: ax.set_title(kwargs_obj.title)

        if kwargs_obj.isocontours != None:
            if np.all(kwargs_obj.isocontour_field) == None: kwargs_obj.isocontour_field = field
            kwargs_isocontour = dict(kwargs, do_log=False)
            _, _, _, _, isocontour_field2d = self.get_field2d(kwargs.isocontour_field, **kwargs_isocontour)
            ax.contour(coord12d / const.kpc, coord22d / const.kpc, isocontour_field2d, levels=kwargs_obj.isocontours, colors=kwargs_obj.color_isocontour, linewidths=2, linestyles='solid')

        if kwargs.plot_star:
            in_slice_star = np.abs(self.coord_star[slice] - kwargs_obj.slice_coord) < self.dx
            ax.plot(self.coord_star[coord1_idx][in_slice_star] / const.kpc, self.coord_star[coord2_idx][in_slice_star] / const.kpc, marker='*', color=kwargs_obj.color_star, linestyle='')

        if kwargs.plot_dm:
            in_slice_dm = np.abs(self.coord_dm[slice] - kwargs.slice_coord) < self.dx
            ax.plot(self.coord_dm[coord1_idx][in_slice_dm] / const.kpc, self.coord_dm[coord2_idx][in_slice_dm] / const.kpc, marker='.', color=kwargs_obj.color_dm, linestyle='')

        return im
    
    def plot_rgb_on_ax(self, ax, field_red, field_green, field_blue, extrema, xlabels=[None, None, None], **kwargs):
        
        kwargs_obj = SimpleNamespace(**kwargs)

        _, _, coord12d, coord22d, field_red2d = self.get_field2d(field_red, **kwargs)
        _, _, _, _, field_green2d = self.get_field2d(field_green, **kwargs)
        _, _, _, _, field_blue2d = self.get_field2d(field_blue, **kwargs)

        log_norm = mpl.colors.LogNorm(vmin=kwargs_obj.extrema[0], vmax=kwargs_obj.extrema[1], clip=True)
        lin_norm = mpl.colors.Normalize(vmin=np.log10(kwargs_obj.extrema), vmax=np.log10(kwargs_obj.extrema[1]), clip=True)

        red = log_norm(field_red2d).data
        green = log_norm(field_green2d).data
        blue = log_norm(field_blue2d).data

        color = np.array([red, green, blue])
        color = np.moveaxis(color, 0, -1)
        plt.pcolormesh(coord12d / const.kpc, coord22d / const.kpc, color)

        if kwargs_obj.do_cbar: 
            
            assert len(xlabels) == 3, "xlabels list must have length 3."

            for i in range(3):
                sm = mpl.cm.ScalarMappable(cmap=[rgb_red, rgb_green, rgb_blue][i], norm=lin_norm)    
                sm.set_array([])
                if i < 2: 
                    self.plot_cbar(ax, sm, extrema, kwargs_obj.cbar_tick_increment, None, "vertical", xlabel=xlabels[i], do_ticks=False)
                else:
                    self.plot_cbar(ax, sm, extrema, kwargs_obj.cbar_tick_increment, kwargs_obj.cbar_label, "vertical", xlabel=xlabels[i], do_ticks=True)

    
    def plot_AH_on_ax(self, ax, field2d, extrema, unit=1., do_log=True, nlevels=200, cmap='jet', cbar_label='field', cbar_tick_increment=None, num_axis_lines=12, axis_labels=True, do_xlabel=True, do_ylabel=True, do_cbar=True, cbar_orientation='vertical', title=None):
        '''
        Plot a spherical slice of a field using the Aitoff-Hammer projection.

        Args
        ax: axis object
        field2d: two-dimensional field, with theta on the first axis and phi on the second
        extrema (tuple): tuple of min and max field value
        unit (float): unit of the field
        do_log (bool): plot the log of the field
        nlevels (int): number of levels in the contour plot
        cmap (str): colormap for the plot
        cbar_label (string): colorbar label
        cbar_tick_increment (float): increment of the colorbar ticks
        num_axis_lines (int): number of lines of constant theta and phi to mark on plot
        axis_labels (bool): label lines of constant theta and phi
        do_xlabel (bool): label x-axis
        do_ylabel (bool): label y-axis
        do_cbar (bool): plot a colorbar
        cbar_orientation (str): orientation of the colorbar
        title (str): title of the plot

        Returns
        im: QuadContourSet object
        '''
        field2d = np.copy(field2d / unit)
        
        ax.set_aspect(3/2)
        ax.axis("off")

        if cbar_tick_increment == None: cbar_tick_increment = (extrema[1] - extrema[0]) / 10

        if do_log: 
            field2d = np.log10(field2d + epsilon)
            extrema = (np.log10(extrema[0]), np.log10(extrema[1]))

        for i, axis_line_value in enumerate(np.arange(0, (num_axis_lines + 0.5) * np.pi / num_axis_lines, np.pi / num_axis_lines)):

            AH1_h_axis_line, AH2_h_axis_line = calc_AH_coords(axis_line_value * np.ones_like(self.coord1d_sph[PH]), self.coord1d_sph[PH])
            ax.plot(AH1_h_axis_line, AH2_h_axis_line, color='black', lw=0.75)
            AH1_ph_axis_line, AH2_ph_axis_line = calc_AH_coords(self.coord1d_sph[H], 2 * axis_line_value * np.ones_like(self.coord1d_sph[H]))
            ax.plot(AH1_ph_axis_line, AH2_ph_axis_line, color='black', lw=0.75)

            if axis_labels and i not in [0, num_axis_lines]:

                h_label = r'$%d^\circ$' % int(np.round(axis_line_value * 180 / np.pi))
                ax.annotate(h_label, (AH1_h_axis_line[0], AH2_h_axis_line[0]), xytext=(-3., 0), xycoords='data', textcoords='offset fontsize', rotation=0)
                ph_label = r'$%d^\circ$' % int(np.round(axis_line_value * 360 / (np.pi)))
                ax.annotate(ph_label, (AH1_ph_axis_line[self.N//2], AH2_ph_axis_line[self.N//2]), xytext=(0.3, -0.5), xycoords='data', textcoords='offset fontsize', rotation=90)

        if do_xlabel: ax.annotate(r'$\theta$', (0., 0.5), xytext=(-2.5, -0.25), xycoords='axes fraction', fontsize=20, textcoords='offset fontsize')
        if do_ylabel: ax.annotate(r'$\varphi$', (0.5, 0.), xytext=(-0.5, -1), xycoords='axes fraction', fontsize=20, textcoords='offset fontsize')

        im = ax.contourf(self.coord_AH_at_sph[0], self.coord_AH_at_sph[1], field2d, levels=np.linspace(extrema[0], extrema[1], nlevels), cmap=cmap, extend='both')
        if do_cbar: 
            if cbar_orientation == "vertical":
                cbar_size, cbar_pad = "3%", -0.8
            elif cbar_orientation == "horizontal":
                cbar_size, cbar_pad = "5%", 0.1
            self.plot_cbar(ax, im, extrema, cbar_tick_increment, cbar_label, cbar_orientation, size=cbar_size, pad=cbar_pad)

        if title != None: ax.set_title(title)

        return im
    
    def plot_cbar(self, ax, im, extrema, tick_increment, label, orientation, size="5%", pad=0.05, xlabel=None, do_ticks=True):
        '''
        Plot a colorbar.

        Args
        ax: axis object
        im: QuadContourSet object
        extrema (tuple): tuple of min and max field value
        tick_increment (float): increment of the colorbar ticks
        label (string): colorbar label
        orientation (str): orientation of the colorbar
        size (str): size of the colorbar
        pad (float): padding of the colorbar
        xlabel (str): colorbar label on the x-axis
        do_ticks (bool): plot ticks on the colorbar
        '''
        divider = make_axes_locatable(ax)
        if orientation == "horizontal":
            cax = divider.append_axes("top", size=size, pad=pad)
            plt.colorbar(im, ax=ax, cax=cax, ticks=np.arange(extrema[0], extrema[1] + 0.5 * tick_increment, tick_increment), label=label, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
        elif orientation == "vertical":
            cax = divider.append_axes("right", size=size, pad=pad)
            plt.colorbar(im, ax=ax, cax=cax, ticks=np.arange(extrema[0], extrema[1] + 0.5 * tick_increment, tick_increment), label=label, orientation="vertical")
            if xlabel != None: cax.set_xlabel(xlabel)
    
    def calc_radial_profile(self, field, r_lim=None, nbins=100, weight=None, cond=None):
        '''
        Compute radial profile of field averaged over spherical shells.

        Args
        field: field
        r_lim (float): maximum radius
        nbins (int): number of radial bins
        weight: weight array
        cond: conditional array to select a specific region

        Returns
        r_1d: array of radial bins
        field_1d: array of spherically averaged field values
        '''
        if (r_lim == None) or (r_lim > self.box_size * self.length_unit / 2):
            r_lim = self.box_size * self.length_unit / 2

        cond_sph = self.coord_sph[R] < r_lim
        if np.all(cond) == None: cond = 1.
        if np.all(weight) == None: weight = 1.

        counts1, bins = np.histogram(self.coord_sph[R][cond_sph].flatten(), weights=(field * weight * cond)[cond_sph].flatten(), bins=nbins)
        counts2, bins = np.histogram(self.coord_sph[R][cond_sph].flatten(), weights=(np.ones_like(field) * weight * cond)[cond_sph].flatten(), bins=nbins)

        r_1d = bins[:-1] + np.diff(bins)[0]
        field_1d = counts1 / counts2

        return r_1d, field_1d


    def calc_phase(self, field1, field2, extrema1, extrema2, do_log1=True, do_log2=True, cond=None, nbins=30):
        '''
        Compute distribution of mass in a two-dimensional phase space.

        Args
        field1, field2: fields
        extrema1, extrema2: tuples of max and min field values
        do_log1, do_log2 (bool): space bins logorithmically
        cond: conditional array to select a specific region
        nbins (int): number of bins

        Returns
        field1_2d, field2_2d: array of field bins
        mass_2d: array of mass per bin
        '''
        if do_log1: 
            field1 = np.log10(field1)
            extrema1 = (np.log10(extrema1[0]), np.log10(extrema1[1]))

        if do_log2:
            field2 = np.log10(field2)
            extrema2 = (np.log10(extrema2[0]), np.log10(extrema2[1]))

        if np.all(cond) == None: cond = 1.

        hist, x_bins, y_bins = np.histogram2d(field1.flatten(), field2.flatten(), weights=(self.density * self.dV * cond).flatten(), bins=(nbins, nbins), range=[extrema1, extrema2])

        if do_log1:
            field1_2d = 10**(x_bins[:-1] + np.diff(x_bins)[0])
        else:
            field1_2d = x_bins[:-1] + np.diff(x_bins)[0]

        if do_log2:
            field2_2d = 10**(y_bins[:-1] + np.diff(y_bins)[0])
        else:
            field2_2d = y_bins[:-1] + np.diff(y_bins)[0]

        mass_2d = hist.T

        return field1_2d, field2_2d, mass_2d


    def calc_mean(self, field, weight=None, cond=None, axis=None, do_sum=False):
        '''
        Calculate the mean value of a field

        Args
        field: field
        weight: weight array
        cond: conditional array to select a specific region
        axis: axes over which to compute mean
        do_sum (bool): compute the sum instead of the mean

        Returns
        mean: mean value of the field
        '''
        if np.all(cond) == None: cond = 1.
        if np.all(weight) == None: weight = 1.

        if do_sum:
            mean = np.sum(field * weight * cond, axis=axis)
        else:    
            mean = np.sum(field * weight * cond, axis=axis) / np.sum(np.ones_like(field) * weight * cond, axis=axis)

        return mean
    
    def grad(self, field):
        ''' Compute the gradient of a scalar field. '''
        grad_Phi = np.zeros((3,) + field.shape)
        grad_Phi[X] = np.gradient(field, self.dx, axis=X)
        grad_Phi[Y] = np.gradient(field, self.dx, axis=Y)
        grad_Phi[Z] = np.gradient(field, self.dx, axis=Z)
        return grad_Phi

    def div(self, field):
        ''' Compute the divergence of a vector field. '''
        div_field = np.gradient(field[X], self.dx, axis=X) + np.gradient(field[Y], self.dx, axis=Y) + np.gradient(field[Z], self.dx, axis=Z)
        return div_field

    def curl(self, field):
        ''' Compute the curl of a vector field. '''
        curl_field = np.zeros_like(field)
        curl_field[X] = np.gradient(field[Z], self.dx, axis=Y) - np.gradient(field[Y], self.dx, axis=Z)
        curl_field[Y] = np.gradient(field[X], self.dx, axis=Z) - np.gradient(field[Z], self.dx, axis=X)
        curl_field[Z] = np.gradient(field[Y], self.dx, axis=X) - np.gradient(field[X], self.dx, axis=Y)
        return curl_field
    
    def laplace(self, field):
        ''' Compute the Laplacian of a vector field. '''
        laplace_field = self.div(self.grad(field))
        return laplace_field
    
    def on_face(self, field):
        ''' Linearly interpolate the value of a field to cell faces. '''
        field_on_xface = np.zeros((3, self.N+1, self.N, self.N))
        field_on_yface = np.zeros((3, self.N, self.N+1, self.N))
        field_on_zface = np.zeros((3, self.N, self.N, self.N+1))

        field_on_xface[:, 1:-1, :, :] = (field[:, 1:, :, :] + field[:, :-1, :, :]) / 2
        field_on_xface[:, 0, :, :] = field[:, 0, :, :]
        field_on_xface[:, -1, :, :] = field[:, -1, :, :]

        field_on_yface[:, :, 1:-1, :] = (field[:, :, 1:, :] + field[:, :, :-1, :]) / 2
        field_on_yface[:, :, 0, :] = field[:, :, 0, :]
        field_on_yface[:, :, -1, :] = field[:, :, -1, :]

        field_on_zface[:, :, :, 1:-1] = (field[:, :, :, 1:] + field[:, :, :, :-1]) / 2
        field_on_zface[:, :, :, 0] = field[:, :, :, 0]
        field_on_zface[:, :, :, -1] = field[:, :, :, -1]

        field_on_face = [field_on_xface, field_on_yface, field_on_zface]
        return field_on_face

    
    @cached_property
    def coord_cyl_at_cart(self):
        ''' Cylindrical coordinates on the Cartesian grid.'''
        return coord_conv(self.coord, sys1=CART, sys2=CYL)
    
    @cached_property
    def coord_sph_at_cart(self):
        ''' Cylindrical coordinates on the Cartesian grid.'''
        return coord_conv(self.coord, sys1=CART, sys2=SPH)
    
    @cached_property
    def coord_cart_at_sph(self):
        ''' Cartesian coordinates on the spherical grid.'''
        return coord_conv(self.coord_sph, sys1=SPH, sys2=CART)
    
    @cached_property
    def coord_cyl_at_sph(self):
        ''' Cylindrical coordinates on the spherical grid.'''
        return coord_conv(self.coord_sph, sys1=SPH, sys2=CYL)
    
    @cached_property
    def coord_sph_dm(self):
        ''' Spherical coordinates of dark matter particles.'''
        return coord_conv(self.coord_dm, sys1=CART, sys2=SPH)
    
    @cached_property
    def coord_sph_star(self):
        ''' Spherical coordinates of star particles.'''
        return coord_conv(self.coord_star, sys1=CART, sys2=SPH)
    
    @cached_property
    def vel_vec_sph_at_cart(self):
        ''' Velocity vector in spherical coordinates on the Cartesian grid '''
        return vec_conv(self.vel_vec, self.coord, sys1=CART, sys2=SPH)
    
    @cached_property
    def n_H(self):
        ''' Hydrogen number density '''
        return const.X_cosmo * self.density / const.m_H
    
    @cached_property
    def density_dust(self):
        ''' Dust density '''
        density_dust = self.metallicity * self.density * (1 - self.ion_frac)
        return density_dust
    
    @cached_property
    def density_star(self):
        ''' Density of stellar mass '''
        edges1d = np.zeros((3, self.N+1))
        edges1d[:, :-1] = self.coord1d - self.dx/2
        edges1d[:, -1] = self.coord1d[:, -1] + self.dx/2
        mass_star_grid, _ = np.histogramdd(self.coord_star.T, bins=edges1d, weights=self.mass_star)
        density_star = mass_star_grid / self.dV
        return density_star
    
    @cached_property
    def density_dm(self):
        ''' Density of dark matter '''
        edges1d = np.zeros((3, self.N+1))
        edges1d[:, :-1] = self.coord1d - self.dx/2
        edges1d[:, -1] = self.coord1d[:, -1] + self.dx/2
        mass_dm_grid, _ = np.histogramdd(self.coord_dm.T, bins=edges1d, weights=self.mass_dm)
        density_dm = mass_dm_grid / self.dV
        return density_dm
    
    @cached_property
    def vel_turb(self):
        ''' Turbulent velocity '''
        return np.sqrt(2 * self.energy_turb)
    
    @cached_property
    def temp(self):
        ''' Temperature '''
        return self.pressure / self.density * const.m_H / const.k_B

    @cached_property
    def vel(self):
        ''' Magnitude of the velocity '''
        return norm(self.vel_vec)
    
    @cached_property
    def ang_mom(self):
        ''' Angular momentum '''
        return np.cross(self.coord, self.vel_vec, axis=0)
    
    @cached_property
    def c_s(self):
        ''' Sound speed '''
        return np.sqrt(self.gamma * self.pressure / self.density)
    
    @cached_property
    def mach(self):
        ''' Mach number '''
        return self.vel / self.c_s
    
    @cached_property
    def mach_turb(self):
        ''' Turbulent mach number '''
        return self.vel_turb / self.c_s / np.sqrt(3)
    
    @cached_property
    def alpha_vir(self):
        ''' Virial parameter '''
        return 15 / np.pi * self.c_s**2 * (1 + self.mach_turb**2) / (const.G * self.density * self.dA)
    
    @cached_property
    def ion_frac(self):
        ''' 
        Ionization fraction (n_HII / n_H).
        
        Estimated by solving the Saha equation.
        '''
        lamb_dB = np.sqrt(const.h**2 / (2 * np.pi * const.m_e * const.k_B * self.temp)) # thermal deBroglie wavelength
        g0, g1 = 2, 1 # statistical weights
        A = np.ones_like(self.temp)
        B = (2 / (self.n_H * lamb_dB**3)) * (g1 / g0) * np.exp(-const.energy_HII / (const.k_B * self.temp))
        C = -B
        ion_frac = solve_quadratic(A, B, C)[1]
        ion_frac[ion_frac > 1] = 1.
        ion_frac[ion_frac > 1] = 1.
        return ion_frac
    
    @cached_property
    def SFR_density(self):
        '''
        Star formation rate density.

        For the multi-fallback model (epsilon_SF = None), see Federrath&Klessen2012 (https://arxiv.org/pdf/1209.2856.pdf) and Kretschmer&Teyssier2021 (https://arxiv.org/pdf/1906.11836.pdf) for details.
        The turbulence forcing parameter b varies smoothly between b ~ 1/3 for purely solenoidal (divergence-free) forcing 
        and b ~ 1 for purely compressive (curl-free) forcing of the turbulence.
        A stochastic mixture of forcing modes in 3-d space leads to b ~ 0.4.
        '''
        t_ff = np.sqrt(3 * np.pi / (32 * const.G * self.density)) # free-fall time
        if self.epsilon_SF == None:
            b = 0.4 # turbulence forcing parameter
            s_crit = np.log(self.alpha_vir * (1 + (2 * self.mach_turb**4) / (1 + self.mach_turb**2))) # lognormal critical density for star formation
            sigma_s = np.sqrt(np.log(1 + b**2 * self.mach_turb**2)) # standard deviation of the lognormal subgrid density distribution
            self.epsilon_SF = 1/2 * np.exp(3/8 * sigma_s**2) * (1 + erf((sigma_s**2 - s_crit) / np.sqrt(2 * sigma_s**2))) # star formation efficiency
            SFR_density = self.epsilon_SF * self.density / t_ff 
        else:
            SFR_density = self.epsilon_SF * self.density / t_ff
            SFR_density[self.alpha_vir < self.alpha_vir_crit] = 0.
        return SFR_density
    
    @property
    def summary_stats(self):
        ''' Calculate summary statistics in the regions defined by cond_hydro, cond_star, and cond_dm. '''
        summary_stats = [
            Stat(self.density, "density", 1., "g/cm^3", None, self.cond_hydro),
            Stat(self.temp, "temperature", 1., "K", self.density, self.cond_hydro),
            Stat(self.ion_frac, "ionization frac", 1., "", self.density, self.cond_hydro),
            Stat(self.metallicity, "metallicity", const.Z_sol, "Z_sol", self.density, self.cond_hydro),
            Stat(self.mach, "mach number", 1., "", self.density, self.cond_hydro),
            Stat(self.mach_turb, "turbulent mach number", 1., "", self.density, self.cond_hydro),
            Stat(self.age_star, "star age", const.Myr, "Myr", self.mass_star, self.cond_star),
            Stat(len(self.mass_star[self.cond_star])*1., "star part number", 1., "", None, self.cond_star, is_array=False),
            Stat(len(self.mass_dm[self.cond_dm])*1., "DM part number", 1., "", None, self.cond_dm, is_array=False),
            Stat(self.contamination_frac, "contamination fraction", 1., "", None, self.cond_dm, is_array=False)
        ]
        return summary_stats

    def print_stats(self):             
        ''' Print summary statistics in a nice table. '''         
        table = []
        for stat in self.summary_stats:
            if stat.is_array:
                table.append([stat.name, "%.3g" % stat.max, "%.3g" % stat.min, "%.3g" % stat.mean, stat.unit_name])
            else:
                table.append([stat.name, "", "", "%.3g" % stat.field, stat.unit_name])
        print(tabulate(table, headers=['Field', 'Max', 'Min', 'Mean/Value', 'Unit'], numalign="right"))
    
    @property
    def contamination_frac(self):
        ''' Calculate the contamination fraction i.e. the mass fraction of dark matter particles above the minimum dark particle mass. '''
        min_mass_dm = np.min(self.mass_dm)
        cond_not_min_mass_dm = self.mass_dm > min_mass_dm
        contamination_frac = np.sum(self.mass_dm * cond_not_min_mass_dm * self.cond_dm) / np.sum(self.mass_dm * self.cond_dm)
        return contamination_frac
    
    
class Stat(object):
    ''' 
    Statistic object. 
    
    Args
    field: field
    name (str): name of the field
    unit (float): unit of the field
    unit_name (str): name of the unit of the field
    weight: weight for the computation of the mean
    cond: conditional array to select a specific region
    is_array (bool): field is an array, rather than a single value
    
    Attrs
    field: field
    name (str): name of the field
    unit (float): unit of the field
    unit_name (str): name of the unit of the field
    weight: weight for the computation of the mean
    cond: conditional array to select a specific region
    is_array (bool): field is an array, rather than a single value
    
    min (float): minimum value (property)
    max (float): maximum value (property)
    mean (float): mean value (property)
    summary (dict): summary of statistic (property)
    '''
    def __init__(self, field, name, unit, unit_name, weight, cond, is_array=True):
        
        self.field = field / unit
        self.name = name
        self.unit = unit
        self.unit_name = unit_name
        if np.all(weight) == None: 
             self.weight = np.ones_like(self.field)
        else:
             self.weight = weight
        if np.all(cond) == None: 
            self.cond = np.ones_like(self.field, dtype=bool)
        else:
            self.cond = cond
        self.is_array = is_array
        
    @property
    def min(self):
        ''' Minimum value. '''
        return np.min(self.field[self.cond])
                 
    @property
    def max(self):
        ''' Maximum value. '''
        return np.max(self.field[self.cond])
                 
    @property
    def mean(self):
        ''' Mean value. '''
        return np.sum(self.field * self.weight * self.cond) / np.sum(self.weight * self.cond)
