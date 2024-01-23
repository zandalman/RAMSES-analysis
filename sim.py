import os
from tabulate import tabulate
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interpn
from functools import cached_property
import const
from config import *
from functions import *

# define custom colormaps
varying_color = np.array([np.linspace(0, 1, 256)] * 3).T
const_color = np.array([[0, 0, 0], [1, 0, 0]])
cmap_red = dict(red=varying_color, green=const_color, blue=const_color)
cmap_green = dict(red=const_color, green=varying_color, blue=const_color)
cmap_blue = dict(red=const_color, green=const_color, blue=varying_color)
rgb_red = LinearSegmentedColormap('rgb_red', segmentdata=cmap_red)
rgb_green = LinearSegmentedColormap('rgb_green', segmentdata=cmap_green)
rgb_blue = LinearSegmentedColormap('rgb_blue', segmentdata=cmap_blue)

class Arglist(object):
    ''' Class for lists of function arguments. '''
    def __init__(self, args):
        self.args = np.array(args)

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
    aexp (float): expansion factor
    redshift (float): redshift
    time_now (float): proper time
    universe_age (float): age of the Universe (i.e. proper time at aexp = 1)
    
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
    tau_starbirth: star particle conformal birth times
    time_starbirth: star particle birth times
    age_star: star particle ages
    id_star: star particle ids
    
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
    b_turb: turbulence forcing parameter (cached property)
    
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
        self.sim_dir = move_to_sim_dir(self.sim_round, self.sim_name)
        self.save_dir = os.path.join(save_dir, "round%d" % self.sim_round, self.sim_name)
        
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
            
        self.redshift = 1 / self.aexp - 1
        self.H = self.H0 * np.sqrt(self.Omega_m0 / self.aexp**3 + self.Omega_k0 / self.aexp**2 + self.Omega_L0)
        self.rho_crit = 3 * self.H**2 / (8 * np.pi * const.G)
        self.time_now = self.aexp_to_proper_time(self.aexp)
        self.universe_age = self.aexp_to_proper_time(1.)
        
        self.time_starbirth = self.tau_starbirth / self.H0 + self.universe_age
        self.age_star = self.time_now - self.time_starbirth
        
        self.cond_hydro = np.ones_like(self.density, dtype=bool)
        self.cond_dm = np.ones_like(self.mass_dm, dtype=bool)
        self.cond_star = np.ones_like(self.mass_star, dtype=bool)

        self.create_sph_grid()
        
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
    
    def aexp_to_proper_time(self, a):
        return aexp_to_proper_time(a, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def aexp_to_conformal_time(self, a):
        return aexp_to_conformal_time(a, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def proper_time_to_aexp(self, t):
        return proper_time_to_aexp(t, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def conformal_time_to_aexp(self, tau):
        return conformal_time_to_aexp(tau, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

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
    
    def plot_grid(self, *args, figsize=None, nrows=1, ncols=1, sharex=False, sharey=False, hspace=None, wspace=None, share_cbar=False, plot_type="slice", do_axes_labels=True, **kwargs):
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
        do_axis_labels (bool): use axis labels
        '''
        kwargs = dict(defaults, **kwargs)
        kwargs_obj = SimpleNamespace(**kwargs)
        
        _, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=False)

        func_dict = dict(slice=self.plot_slice_on_ax, rgb_slice=self.plot_rgb_on_ax, AH=self.plot_AH_on_ax)
        
        for arg in args:
            if type(arg) == Arglist:
                assert arg.args.shape[:2] == (nrows, ncols), "Arglist shape must match (nrows, ncols)"
        
        for kwarg_value in kwargs.values():
            if type(kwarg_value) == Arglist:
                assert kwarg_value.args.shape[:2] == (nrows, ncols), "Arglist shape must match (nrows, ncols)"
        
        for i, axs1d in enumerate(axs):
            for j, ax in enumerate(axs1d):
                
                if do_axes_labels:
                    do_xlabel = (i == len(axs1d)-1) if sharex else True
                    do_ylabel = (j == 0) if sharey else True
                else:
                    do_xlabel = do_ylabel = 0

                current_args, current_kwargs = self.get_current_args(args, kwargs, i, j)
                im = func_dict[plot_type](ax, *current_args, **current_kwargs, do_xlabel=do_xlabel, do_ylabel=do_ylabel, do_cbar=(not share_cbar))

        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        if share_cbar: 
            if kwargs_obj.do_log: kwargs_obj.extrema = (np.log10(kwargs_obj.extrema[0]), np.log10(kwargs_obj.extrema[1]))
            plt.colorbar(im, ax=axs, ticks=np.arange(kwargs_obj.extrema[0], kwargs_obj.extrema[1] + 0.5 * kwargs_obj.cbar_tick_increment, kwargs_obj.cbar_tick_increment), label=kwargs_obj.cbar_label, fraction=0.015, pad=0.02)
    
    def plot(self, *args, figsize=None, plot_type="slice", **kwargs):
        '''
        Wrapper function for plot_slice_on_ax, for a single plot.

        Args
        *args: arguments to be passed to plotting function
        **kwargs: keyword arguments to be passed to plotting function
        figsize: figure size
        plot_type (str): plot type ['slice', 'rgb_slice', 'AH']

        Returns
        ax
        '''
        kwargs = dict(defaults, **kwargs)
        
        plt.figure(figsize=figsize)
        ax = plt.gca()
        func_dict = dict(slice=self.plot_slice_on_ax, rgb_slice=self.plot_rgb_on_ax, AH=self.plot_AH_on_ax)
        func_dict[plot_type](ax, *args, **kwargs, do_xlabel=True, do_ylabel=True, do_cbar=True)
        return ax
    
    def get_field_input(self, field, field_name='Field'):
        ''' Return a field given an input, which can either be a field or a string. '''
        if type(field) in [str, np.str_]:
            assert hasattr(self, field), "%s must be an attribute of Sim object if passed as string." % field_name
            field = getattr(self, field)
        return np.copy(field)
    
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
        
        field = self.get_field_input(field, "Field") / kwargs_obj.unit
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
        if kwargs_obj.do_log: extrema = (np.log10(extrema[0]), np.log10(extrema[1]))

        if kwargs_obj.cbar_tick_increment == None: kwargs_obj.cbar_tick_increment = (extrema[1] - extrema[0]) / 10
        
        im = ax.contourf(coord12d / const.kpc, coord22d / const.kpc, field2d, extend='both', cmap=kwargs_obj.cmap, levels=np.linspace(extrema[0], extrema[1], kwargs_obj.nlevels))
        ax.set_aspect(True)
        if kwargs_obj.do_cbar: self.plot_cbar(ax, im, extrema, kwargs_obj.cbar_tick_increment, kwargs_obj.cbar_label, kwargs_obj.cbar_orientation)

        coord_labels = [r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]"]
        coord1_label, coord2_label = coord_labels[coord1_idx], coord_labels[coord2_idx]
        if kwargs_obj.do_xlabel: ax.set_xlabel(coord1_label)
        if kwargs_obj.do_ylabel: ax.set_ylabel(coord2_label)

        if kwargs_obj.width != None:
            ax.set_xlim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)
            ax.set_ylim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)

        if kwargs_obj.title != None: ax.set_title(kwargs_obj.title)

        if kwargs_obj.isocontours != None:
            if np.all(kwargs_obj.isocontour_field) == None: kwargs_obj.isocontour_field = field
            kwargs_isocontour = dict(kwargs, do_log=False)
            _, _, _, _, isocontour_field2d = self.get_field2d(kwargs_obj.isocontour_field, **kwargs_isocontour)
            ax.contour(coord12d / const.kpc, coord22d / const.kpc, isocontour_field2d, levels=kwargs_obj.isocontours, colors=kwargs_obj.color_isocontour, linewidths=2, linestyles='solid')

        if kwargs_obj.plot_star:
            in_slice_star = np.abs(self.coord_star[slice] - kwargs_obj.slice_coord) < self.dx
            ax.plot(self.coord_star[coord1_idx][in_slice_star] / const.kpc, self.coord_star[coord2_idx][in_slice_star] / const.kpc, marker='*', color=kwargs_obj.color_star, linestyle='')

        if kwargs_obj.plot_dm:
            in_slice_dm = np.abs(self.coord_dm[slice] - kwargs_obj.slice_coord) < self.dx
            ax.plot(self.coord_dm[coord1_idx][in_slice_dm] / const.kpc, self.coord_dm[coord2_idx][in_slice_dm] / const.kpc, marker='.', color=kwargs_obj.color_dm, linestyle='')

        return im
    
    def plot_rgb_on_ax(self, ax, field_red, field_green, field_blue, extrema, xlabels=[None, None, None], **kwargs):
        '''
        Plot the cross section of a field perpendicular to a coordinate axis.

        Args
        ax: axis object
        field_red, field_green, field_blue: fields to map to red, green, and blue
        extrema (tuple): tuple of min and max field value
        xlabels (list): list of xlabels of the RGB colorbars
        kwargs: keyword arguments
            width (float): width of plot
            cbar_label (string): colorbar label
            cbar_tick_increment (float): increment of the colorbar ticks
            do_xlabel (bool): label x-axis
            do_ylabel (bool): label y-axis
            do_cbar (bool): plot a colorbar
            title (str): title of the plot
            keyword arguments of self.get_field2d
        '''
        kwargs_obj = SimpleNamespace(**kwargs)

        coord1_idx, coord2_idx, coord12d, coord22d, field_red2d = self.get_field2d(field_red, **kwargs)
        _, _, _, _, field_green2d = self.get_field2d(field_green, **kwargs)
        _, _, _, _, field_blue2d = self.get_field2d(field_blue, **kwargs)

        if kwargs_obj.do_log: extrema = (np.log10(extrema[0]), np.log10(extrema[1]))
        norm = mpl.colors.Normalize(vmin=extrema[0], vmax=extrema[1], clip=True)

        if kwargs_obj.cbar_tick_increment == None: kwargs_obj.cbar_tick_increment = (extrema[1] - extrema[0]) / 10

        red = norm(field_red2d).data
        green = norm(field_green2d).data
        blue = norm(field_blue2d).data

        color = np.array([red, green, blue])
        color = np.moveaxis(color, 0, -1)
        plt.pcolormesh(coord12d / const.kpc, coord22d / const.kpc, color)

        coord_labels = [r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]"]
        coord1_label, coord2_label = coord_labels[coord1_idx], coord_labels[coord2_idx]
        if kwargs_obj.do_xlabel: ax.set_xlabel(coord1_label)
        if kwargs_obj.do_ylabel: ax.set_ylabel(coord2_label)

        if kwargs_obj.width != None:
            ax.set_xlim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)
            ax.set_ylim(-kwargs_obj.width / const.kpc, kwargs_obj.width / const.kpc)

        if kwargs_obj.title != None: ax.set_title(kwargs_obj.title)

        if kwargs_obj.do_cbar: 
        
            assert len(xlabels) == 3, "xlabels list must have length 3."
            divider = make_axes_locatable(ax)

            for i in range(3):
                sm = mpl.cm.ScalarMappable(cmap=[rgb_red, rgb_green, rgb_blue][i], norm=norm)    
                sm.set_array([])
                if i < 2: 
                    self.plot_cbar(ax, sm, extrema, kwargs_obj.cbar_tick_increment, None, "vertical", xlabel=xlabels[i], do_ticks=False, size="5%", pad=0.1, divider=divider)
                else:
                    self.plot_cbar(ax, sm, extrema, kwargs_obj.cbar_tick_increment, kwargs_obj.cbar_label, "vertical", xlabel=xlabels[i], do_ticks=True, size="5%", pad=0.1, divider=divider)

        return sm
    
    def plot_AH_on_ax(self, ax, field2d, extrema, unit=1., do_log=True, nlevels=200, cmap='jet', cbar_label='field', cbar_tick_increment=None, num_axis_lines=12, axis_labels=True, do_xlabel=True, do_ylabel=True, do_cbar=True, cbar_orientation='vertical', title=None, **kwargs):
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
    
    def plot_cbar(self, ax, im, extrema, tick_increment, label, orientation, size="5%", pad=0.05, xlabel=None, do_ticks=True, divider=None):
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
        divider: axis divider object
        '''
        if divider == None: divider = make_axes_locatable(ax)
        ticks = np.arange(extrema[0], extrema[1] + 0.5 * tick_increment, tick_increment) if do_ticks else []
        if orientation == "horizontal":
            cax = divider.append_axes("top", size=size, pad=pad)
            plt.colorbar(im, ax=ax, cax=cax, ticks=ticks, label=label, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
        elif orientation == "vertical":
            cax = divider.append_axes("right", size=size, pad=pad)
            plt.colorbar(im, ax=ax, cax=cax, ticks=ticks, label=label, orientation="vertical")
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


    def calc_phase(self, *args, **kwargs):
        '''
        Wrapper for calc_phase weighting by mass.
        '''
        weight = self.density * self.dV
        field1_2d, field2_2d, mass_2d = calc_phase(*args, weight=weight, **kwargs)
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
        return np.sqrt(2/3 * self.energy_turb)
    
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
        return self.vel_turb / self.c_s
    
    @cached_property
    def alpha_vir(self):
        ''' Virial parameter '''
        return 15 / np.pi * self.c_s**2 * (1 + self.mach_turb**2) / (const.G * self.density * self.dx_local**2)
    
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
        ''' Star formation rate density. '''
        t_ff = np.sqrt(3 * np.pi / (32 * const.G * self.density)) # free-fall time
        if self.epsilon_SF == None:
            self.epsilon_SF = calc_epsilon_SF(self.alpha_vir, self.mach_turb, b_turb=1.0)
            SFR_density = self.epsilon_SF * self.density / t_ff 
        else:
            SFR_density = self.epsilon_SF * self.density / t_ff
            SFR_density[self.alpha_vir < self.alpha_vir_crit] = 0.
        return SFR_density
    
    @cached_property
    def b_turb(self):
        '''
        Turbulence forcing parameter

        Heuristically combine the divergence and the curl of the velocity
        based on eq. 23 of Federrath+2010 (https://arxiv.org/pdf/0905.1060.pdf)
        '''
        vel_div = self.div(self.vel_vec)
        vel_curl = self.curl(self.vel_vec)
        E_div = norm(vel_div)**2
        E_curl = norm(vel_curl)**2
        zeta = E_div / (E_div + E_curl)
        b_turb = 1/3 + 2/3 * zeta**3
        return b_turb
    
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
