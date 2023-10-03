import os, subprocess, warnings
from datetime import datetime
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datetime import datetime
from scipy.integrate import quad, quad_vec, trapz, cumtrapz
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.interpolate import griddata, interpn

import const

X, Y, Z = 0, 1, 2
R, H, PH = 0, 1, 2
ALPHA_EPS0P01, ALPHA_EPS0P1, ALPHA_EPS1P0, DMO, GAS = 0, 1, 2, 3, 4
HYDRO, DM, STAR = 0, 1, 2
epsilon = 1e-30

analysis_dir = "/home/za9132/analysis"
save_dir = os.path.join(analysis_dir, "figures")
sim_base_dir = "/home/za9132/scratch/romain"
list_of_sim_name = ["alpha_eps0p01", "alpha_eps0p1", "alpha_eps1p0", "dmo", "gas"]
list_of_sim_latex = [r"$\varepsilon_{\rm SF} = 0.01$", r"$\varepsilon_{\rm SF} = 0.1$", r"$\varepsilon_{\rm SF} = 1.0$", "Dark Matter Only", "Multi-Freefall Model"]


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
        plt.savefig(os.path.join(save_dir, "all", filename), bbox_inches="tight", dpi=dpi)
        print("Saved figure as '%s'" % filename)
        
        
def calc_fft(x, y):
    '''
    Compute the Fourier transform of an array.
    
    Args
    x: array of values at which y is sampled, must be uniform
    y: array to Fourier transform
    
    Returns
    x_fft: array of frequencies at which y_fft is sampled
    y_fft: Fourier transform of y
    '''
    y_fft = np.fft.rfft(y)
    x_fft = np.fft.rfftfreq(y.size, d=np.diff(x)[0])
    
    return x_fft, y_fft


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
    
    
class Sim(object):
    '''
    Simulation object.
    
    Args
    sim_idx (int): index of the simulation (ALPHA_EPS0P01, ALPHA_EPS0P1, ALPHA_EPS1P0, DMO, GAS)
    npz_file (str): filename of the npz file with the sim data
    
    Attrs
    
    halo_idx (int): index of the halo
    halo_mass (float): mass of the halo
    a_exp (float): expansion factor
    redshift (float): redshift
    current_time (float): proper time
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
    amr_level (int): number of AMR levels in the simulation
    box_size (float): simulation box size in code units of length
    left_edge: array of coordinate of the left corner of the simulation box in code units of length
    N (int): array size
    dx (float): length element
    dV (float): volume element
    dm_coord: array of dark matter particle coordinates
    dm_mass: array of dark matter particles masses
    star_coord: array of star particle coordinates
    star_mass: array of star particle masses
    star_birth_time: array of star particle birth times
    coord: coordinate array
    coord1d: 1d coordinate array
    density: array of densities
    metallicity: array of metallicities
    pressure: array of pressures
    turb_energy: array of turbulent energies
    refinement_criterion: array of refinement criteria
    vel_vec: array of velocity vectors
    r: radial coordinate array
    r_dm: array of radial coordinates of dark matter particles
    r_star: array of radial coordinates of star matter particles
    temperature: array of temperatures
    vel: array of velocity magnitudes
    
    sim_idx (int): index of the simulation
    sim_name (str): name of the simulation
    sim_latex (str): latex string of the simulation, for plotting
    sim_dir (str): path to the simulation data
    save_dir (str): path of the directory to save figures
    npz_file (str): name of the npz file
    n_H: array of hydrogen number densities
    v_turb: array of turbulent velocities
    star_age: array of ages of star particles
    gamma (float): adiabatic index
    c_s: array of sound speeds
    mach: array of Mach numbers
    v_turb_1d: array of one-dimensional turbulent velocities
    mach_turb: array of turbulent mach numbers
    alpha_vir: array of virial parameters
    t_ff: array of freefall times
    epsilon_SF: array of star formation efficiencies
    SFR_density: array of star formation rate densities
    summury_stats (dict): summary statistics
    '''
    def __init__(self, sim_idx, npz_file, epsilon_SF=None):
        
        self.sim_idx = sim_idx
        self.sim_name = list_of_sim_name[self.sim_idx]
        self.sim_latex = list_of_sim_latex[self.sim_idx]
        self.sim_dir = os.path.join(sim_base_dir, self.sim_name)
        self.save_dir = os.path.join(save_dir, self.sim_name)
        
        os.chdir(self.sim_dir)
        print("Moving to directory '%s'." % self.sim_dir)
        
        self.npz_file = npz_file
        data = np.load(self.npz_file)
        for var_name in data:
            setattr(self, var_name, data[var_name])
            
        self.dA = self.dx**2
        self.coord_r_1d = np.linspace(0, np.max(self.coord1d), self.N // 2)
        self.coord_h_1d = np.linspace(0, np.pi, self.N // 2)
        self.coord_ph_1d = np.linspace(0, 2 * np.pi, self.N)
        self.coord_sph = np.array(np.meshgrid(self.coord_r_1d, self.coord_h_1d, self.coord_ph_1d, indexing='ij'))
        self.dr = np.diff(self.coord_r_1d)[0]
        self.dh = np.diff(self.coord_h_1d)[0]
        self.dph = np.diff(self.coord_ph_1d)[0]
        self.coord_cart_at_sph = np.array([
            self.coord_sph[R] * np.cos(self.coord_sph[PH]) * np.sin(self.coord_sph[H]),
            self.coord_sph[R] * np.sin(self.coord_sph[PH]) * np.sin(self.coord_sph[H]),
            self.coord_sph[R] * np.cos(self.coord_sph[H])
        ])
        self.coord_AH_at_sph = np.array(calc_AH_coords(self.coord_sph[H, 0], self.coord_sph[PH, 0]))
        self.dA_hph = np.sin(self.coord_sph[H]) * self.dh * self.dph

        self.n_H = const.X_cosmo * self.density / const.m_H
        self.v_turb = np.sqrt(2 * self.turb_energy)
        self.star_age = self.current_time - self.star_birth_time
        self.gamma = 5/3
        self.c_s = np.sqrt(self.gamma * self.pressure / self.density)
        self.mach = self.vel / self.c_s
        self.v_turb_1d = self.v_turb / np.sqrt(3)
        self.mach_turb = self.v_turb_1d / self.c_s
        self.alpha_vir = 15 / np.pi * self.c_s**2 * (1 + self.mach_turb**2) / (const.G * self.density * self.dx**2)
        self.t_ff = np.sqrt(3 * np.pi / (32 * const.G * self.density))
        self.n_dust = self.metallicity * self.n_H
        self.n_dust[self.temperature > 1e4] = 0

        if epsilon_SF == None:
            '''
            See Federrath&Klessen2012 (https://arxiv.org/pdf/1209.2856.pdf) and Kretschmer&Teyssier2021 (https://arxiv.org/pdf/1906.11836.pdf) for details.
            b varies smoothly between b ~ 1/3 for purely solenoidal (divergence-free) forcing 
            and b ~ 1 for purely compressive (curl-free) forcing of the turbulence
            A stochastic mixture of forcing modes in 3-d space leads to b ~ 0.4
            '''
            b = 1. # turbulence forcing parameter
            s_crit = np.log(self.alpha_vir * (1 + (2 * self.mach_turb**4) / (1 + self.mach_turb**2))) # lognormal critical density for star formation
            sigma_s = np.sqrt(np.log(1 + b**2 * self.mach_turb**2)) # standard deviation of the lognormal subgrid density distribution
            self.epsilon_SF = 1/2 * np.exp(3/8 * sigma_s**2) * (1 + erf((sigma_s**2 - s_crit) / np.sqrt(2 * sigma_s**2))) # star formation efficiency
            self.SFR_density = self.epsilon_SF * self.density / self.t_ff
        else:
            self.epsilon_SF = epsilon_SF
            self.SFR_density = self.epsilon_SF * self.density / self.t_ff
            # not correctly implemented; need density criterion for SF; why is alpha_vir so big when it is unitless?
            
        self.summary_stats = {}
            
            
    def save_fig(self, fig_name, filetype="png", dpi=300):
        '''
        Save the current matplotlib figure.

        Args
        name (string): figure name
        filetype (string): file type
        dpi (int): dots per inch
        '''
        datetime_string = datetime.now().strftime("%m%d%Y%H%M")
        filename = "%s-%s.%s" % (fig_name, datetime_string, filetype)
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight", dpi=dpi)
        print("Saved figure as '%s'" % filename)
        

    def a_exp_to_proper_time(self, a):
        '''Convert expansion factor to proper time.'''
        integrand = lambda a: (self.Omega_m0 * a**(-1) + self.Omega_k0 + self.Omega_L0 * a**2)**(-1/2)
        t = quad(integrand, 0, a)[0] / self.H0

        return t


    def a_exp_to_conformal_time(self, a):
        '''Convert expansion factor to conformal time.'''
        integrand = lambda a: (self.Omega_m0 * a + self.Omega_k0 * a**2 + self.Omega_L0 * a**4)**(-1/2)
        tau = const.c * quad(integrand, 0, a)[0] / self.H0

        return tau


    def proper_time_to_a_exp(self, t):
        '''Convert proper time to expansion rate.'''
        a = fsolve(lambda a: (a_exp_to_proper_time(a) - t) * self.H0, self.a_exp)

        return a


    def conformal_time_to_a_exp(self, tau):
        '''Convert conformal time to expansion rate.'''
        a = fsolve(lambda a: (a_exp_to_conformal_time(a) - tau) * self.H0, self.a_exp)

        return a


    def interp_to_sph(self, field):
        ''' Interpolate a field to a spherical grid '''
        field_sph = interpn(self.coord1d, field, np.moveaxis(self.coord_cart_at_sph, 0, -1))
        return field_sph
    
    
    def plot_slice(self, field, extrema, slice=Z, project=False, weight=None, cond=None, avg=True, width=None, do_log=True, slice_coord=None, nlevels=200, cmap='jet', cbar_label='field', cbar_tick_increment=None, plot_star=False, plot_dm=False, isocontours=None):
        '''
        Plot a cross section of a field perpendicular to a coordinate axis.

        Args
        field: field
        extrema (tuple): tuple of min and max field value
        slice (int): index of the coordinate direction perpendicular to the slice plane
        project (bool): project the field along the slice direction
        weight: weight array to use for the project
        cond: conditional array to select a specific region for the projection
        avg (bool): take the average (rather than sum) in the projection
        width (float): width of plot
        do_log (bool): plot the log of the field
        slice_coord (float): coordinate of the cross section
        nlevels (int): number of levels in the contour plot
        cmap (str): colormap for the plot
        cbar_label (string): colorbar label
        cbar_tick_increment (float): increment of the colorbar ticks
        plot_star (bool): plot markers for star particles within one cell of the cross section
        plot_dm (bool): plot markers for dark matter particles within one cell of the cross section
        isocontours: list of field values to plot isocontours
        '''
        if np.all(weight) == None: weight = 1.
        if np.all(cond) == None: cond = 1.

        if slice_coord == None:
            slice_coord = 0
            slice_coord_idx = self.N//2
        else:
            slice_coord_idx = np.argmin(np.abs(self.coord1d[slice] - slice_coord))

        if cbar_tick_increment == None: cbar_tick_increment = (extrema[1] - extrema[0]) / 10

        coord1_idx, coord2_idx = np.sort([(slice + 1) % 3, (slice + 2) % 3])

        coord12d = self.coord[coord1_idx].take(indices=slice_coord_idx, axis=slice)
        coord22d = self.coord[coord2_idx].take(indices=slice_coord_idx, axis=slice)

        if project:
            field2d = np.sum(field * weight * cond * self.dx, axis=slice)
            if avg: field2d /= np.sum(np.ones_like(field) * weight * cond * self.dx + epsilon, axis=slice)
        else:
            field2d = field.take(indices=slice_coord_idx, axis=slice)

        if do_log: 
            field2d = np.log10(field2d + epsilon)
            extrema = (np.log10(extrema[0]), np.log10(extrema[1]))
            if isocontours != None:
                isocontours = [np.log10(isocontour) for isocontour in isocontours]

        plt.contourf(coord12d / const.kpc, coord22d / const.kpc, field2d, extend='both', cmap=cmap, levels=np.linspace(extrema[0], extrema[1], nlevels))
        plt.gca().set_aspect(True)
        plt.colorbar(ticks=np.arange(extrema[0], extrema[1] + cbar_tick_increment, cbar_tick_increment), label=cbar_label)

        coord_labels = [r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]"]
        coord1_label, coord2_label = coord_labels[coord1_idx], coord_labels[coord2_idx]
        plt.xlabel(coord1_label); plt.ylabel(coord2_label)

        if width != None:
            plt.xlim(-width / const.kpc, width / const.kpc)
            plt.ylim(-width / const.kpc, width / const.kpc)

        if isocontours != None:
            plt.contour(coord12d / const.kpc, coord22d / const.kpc, field2d, levels=isocontours, colors='black', linewidths=2, linestyles='solid')

        if plot_star:

            in_slice_star = np.abs(self.star_coord[slice] - slice_coord) < self.dx
            plt.plot(self.star_coord[coord1_idx][in_slice_star] / const.kpc, self.star_coord[coord2_idx][in_slice_star] / const.kpc, marker='*', color='black', linestyle='')

        if plot_dm:

            in_slice_dm = np.abs(self.dm_coord[slice] - slice_coord) < self.dx
            plt.plot(self.dm_coord[coord1_idx][in_slice_dm] / const.kpc, self.dm_coord[coord2_idx][in_slice_dm] / const.kpc, marker='.', color='black', linestyle='')


    def plot_AH(self, field2d, extrema, do_log=True, nlevels=200, cmap='jet', cbar_label='field', cbar_tick_increment=None, num_axis_lines=12, axis_labels=True):
        '''
        Plot a spherical slice of a field using the Aitoff-Hammer projection.

        Args
        field2d: two-dimensional field, with theta on the first axis and phi on the second
        extrema (tuple): tuple of min and max field value
        do_log (bool): plot the log of the field
        nlevels (int): number of levels in the contour plot
        cmap (str): colormap for the plot
        cbar_label (string): colorbar label
        cbar_tick_increment (float): increment of the colorbar ticks
        num_axis_lines (int): number of lines of constant theta and phi to mark on plot
        axis_labels (bool): label lines of constant theta and phi
        '''
        plt.gca().set_aspect(3/2)
        plt.axis("off")

        if cbar_tick_increment == None: cbar_tick_increment = (extrema[1] - extrema[0]) / 10

        if do_log: 
            field2d = np.log10(field2d + epsilon)
            extrema = (np.log10(extrema[0]), np.log10(extrema[1]))

        for i, axis_line_value in enumerate(np.arange(0, (num_axis_lines + 0.5) * np.pi / num_axis_lines, np.pi / num_axis_lines)):

            AH1_h_axis_line, AH2_h_axis_line = calc_AH_coords(axis_line_value * np.ones_like(self.coord_ph_1d), self.coord_ph_1d)
            plt.plot(AH1_h_axis_line, AH2_h_axis_line, color='black', lw=0.75)
            AH1_ph_axis_line, AH2_ph_axis_line = calc_AH_coords(self.coord_h_1d, 2 * axis_line_value * np.ones_like(self.coord_h_1d))
            plt.plot(AH1_ph_axis_line, AH2_ph_axis_line, color='black', lw=0.75)

            if axis_labels and i not in [0, num_axis_lines]:

                h_label = r'$%d^\circ$' % int(np.round(axis_line_value * 180 / np.pi))
                plt.annotate(h_label, (AH1_h_axis_line[0], AH2_h_axis_line[0]), xytext=(-3., 0), xycoords='data', textcoords='offset fontsize', rotation=0)
                ph_label = r'$%d^\circ$' % int(np.round(axis_line_value * 360 / (np.pi)))
                plt.annotate(ph_label, (AH1_ph_axis_line[self.N//4], AH2_ph_axis_line[self.N//4]), xytext=(0.3, -0.5), xycoords='data', textcoords='offset fontsize', rotation=90)

        plt.annotate(r'$\theta$', (0., 0.5), xytext=(-2.5, -0.25), xycoords='axes fraction', fontsize=20, textcoords='offset fontsize')
        plt.annotate(r'$\varphi$', (0.5, 0.), xytext=(-0.5, -1), xycoords='axes fraction', fontsize=20, textcoords='offset fontsize')

        plt.contourf(self.coord_AH_at_sph[0], self.coord_AH_at_sph[1], field2d, levels=np.linspace(extrema[0], extrema[1], nlevels), cmap=cmap, extend='both')
        plt.colorbar(ticks=np.arange(extrema[0], extrema[1] + cbar_tick_increment, cbar_tick_increment), label=cbar_label) 
    
    
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

        cond_sph = self.r < r_lim
        if np.all(cond) == None: cond = 1.
        if np.all(weight) == None: weight = 1.

        counts1, bins = np.histogram(self.r[cond_sph].flatten(), weights=(field * weight * cond)[cond_sph].flatten(), bins=nbins)
        counts2, bins = np.histogram(self.r[cond_sph].flatten(), weights=(np.ones_like(field) * weight * cond)[cond_sph].flatten(), bins=nbins)

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


    def calc_mean(self, field, weight=None, cond=None):
        '''
        Calculate the mean value of a field

        Args
        field: field
        weight: weight array
        cond: conditional array to select a specific region

        Returns
        mean: mean value of the field
        '''
        if np.all(cond) == None: cond = 1.
        if np.all(weight) == None: weight = 1.

        mean = np.sum(field * weight * cond) / np.sum(np.ones_like(field) * weight * cond)

        return mean


    def calc_contamination_frac(self, cond=None):
        '''
        Calculate the contamination fraction,
        defined as the mass fraction of dark matter particles above the minimum dark particle mass

        Args
        cond: conditional array to select a specific region

        Returns
        contamination_frac: contamination fraction
        '''
        if np.all(cond) == None: cond = np.ones_like(self.dm_mass)

        min_dm_mass = np.min(self.dm_mass[cond])
        cond_not_min_dm_mass = self.dm_mass > min_dm_mass

        contamination_frac = np.sum(self.dm_mass * cond * cond_not_min_dm_mass) / np.sum(self.dm_mass * cond)

        return contamination_frac


    def calc_summary_stats(self, fields, names, weights, units, types, cond_hydro=None, cond_dm=None, cond_star=None, do_print=True):
        '''
        Calculate summary statistics for a region

        fields: list of fields to compute the min, max, and mean
        names: list of field names, must be the same length as fields
        weights: list of weights to use when computing field means, must be the same length as fields
        units: list of unit strings, must be the same length as fields
        types: list of field types (HYDRO, DM, STAR), must be the same length as fields
        cond_hydro: conditional array to select a specific region for hydro fields
        cond_dm: conditional array to select a specific region for dark matter fields
        cond_star: conditional array to select a specific region for star fields
        do_print (bool): print the summary stats
        '''
        for lst_name, lst in dict(names=names, weigts=weights, units=units, types=types).items():
            assert len(lst) == len(fields), "Length of %s list should match length of fields list" % lst_name

        if np.all(cond_hydro) == None: cond_hydro = (self.density >= 0)
        if np.all(cond_dm) == None: cond_dm = (self.dm_mass >= 0)
        if np.all(cond_star) == None: cond_star = (self.star_mass >= 0)
        
        for i, field in enumerate(fields):

            cond = [cond_hydro, cond_dm, cond_star][types[i]]
            if units[i] == None: units[i] = ""

            min_field = np.min(field[cond])
            max_field = np.max(field[cond])
            mean_field = self.calc_mean(field, weight=weights[i], cond=cond)

            self.summary_stats[names[i]] = dict(min=min_field, max=max_field, mean=mean_field, unit=units[i])

        self.summary_stats["star part number"] = dict(min=0, max=0, mean=len(self.star_mass[cond_star]), unit="")
        self.summary_stats["dark matter part number"] = dict(min=0, max=0, mean=len(self.dm_mass[cond_dm]), unit="")
        self.summary_stats["contamination frac"] = dict(min=0, max=0, mean=self.calc_contamination_frac(cond=cond_dm), unit="")
        
        if do_print:
            table = []
            for field, stats in self.summary_stats.items():
                table.append([field, "%.3g" % stats['max'], "%.3g" % stats['min'], "%.3g" % stats['mean'], stats['unit']])
            print(tabulate(table, headers=['Field', 'Max', 'Min', 'Mean', 'Unit']))

