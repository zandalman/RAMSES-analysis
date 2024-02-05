import os
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property
import const
from config import *
from functions import *

class Sim(object):
    '''
    Simulation object.
    
    Args
    sim_round (int): round of simulation runs
    sim_name (str): name of the simulation
    npz_file (str): filename of the npz file with the simulation data
    
    Attrs
    npz_file (str): filename of the npz file with the simulation data

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
    
    coord_dm: coordinates of dark matter particles
    mass_dm: dark matter particle masses
    
    coord_star: coordinates of star particles
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
    eps_sf: star formation efficiency (cached property)
    t_ff: freefall time (cached property)
    sfr_density: star formation rate density (cached property)
    
    vel_vec: velocity vector
    vel: magnitude of the velocity (cached property)
    vel_turb: turbulent velocity (cached property)
    
    sim_latex (str): latex string of the simulation, for plotting
    sim_dir (str): path to the simulation data
    save_dir (str): path of the directory to save figures
    '''
    def __init__(self, sim_round, sim_name, npz_file):
        
        self.sim_round = sim_round
        self.sim_name = sim_name
        self.sim_latex = sim_name_to_latex[self.sim_name]
        self.sim_dir = move_to_sim_dir(self.sim_round, self.sim_name)
        self.save_dir = os.path.join(save_dir, "round%d" % self.sim_round, self.sim_name)
        
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
            
        if hasattr(self, 'a_exp'): self.aexp = self.a_exp
        self.redshift = 1 / self.aexp - 1
        self.H = self.H0 * np.sqrt(self.Omega_m0 / self.aexp**3 + self.Omega_k0 / self.aexp**2 + self.Omega_L0)
        self.rho_crit = 3 * self.H**2 / (8 * np.pi * const.G)
        self.time_now = self.aexp_to_proper_time(self.aexp)
        self.universe_age = self.aexp_to_proper_time(1.)
        
        self.time_starbirth = self.tau_starbirth / self.H0 + self.universe_age
        self.age_star = self.time_now - self.time_starbirth

    def aexp_to_proper_time(self, a):
        return aexp_to_proper_time(a, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def aexp_to_conformal_time(self, a):
        return aexp_to_conformal_time(a, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def proper_time_to_aexp(self, t):
        return proper_time_to_aexp(t, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def conformal_time_to_aexp(self, tau):
        return conformal_time_to_aexp(tau, self.Omega_m0, self.Omega_k0, self.Omega_L0, self.H0)

    def get_field_input(self, field):
        ''' Return a field given an input, which can either be a field or a string. '''
        if type(field) in [str, np.str_]:
            assert hasattr(self, field), "Field must be an attribute of Sim object if passed as string."
            field = np.copy(getattr(self, field))
        return field
    
    def getimgplane_anyaxis(self, size_img, vec_camera=np.array([0, 0, 1]), vec_north=np.array([0, 1, 0]), coord_center=np.array([0, 0, 0]), num_pxl=128):
        '''
        Get the image plane perpendicular to any axis.
        
        Args
        size_img (float): width of the image [cm]
        vec_camera: vector orieted perpendicular to the projection plane
        vec_north: vector oriented upwards in the image
        coord_center: coordinate of the center of the image plane
        num_pxl (int): width of the image in pixels
        
        Returns
        vec_coord1, vec_coord2: image plane vectors
        coord_pxl: pixel coordinate array
        '''
        if size_img == None: size_img = self.box_size
        
        # compute the image plane vectors
        vec_coord2 = np.cross(vec_north, vec_camera)
        vec_coord2 = vec_coord2 / norm(vec_coord2)
        vec_coord1 = -np.cross(vec_coord2, vec_camera)
        vec_coord1 = vec_coord1 / norm(vec_coord1)
        
        # compute the pixel coordinate array
        arr_pxl = np.linspace(-size_img/2, size_img/2, num_pxl)
        coord_pxl = coord_center[:, None, None] + vec_coord1[:, None, None] * arr_pxl[None, :, None] + vec_coord2[:, None, None] * arr_pxl[None, None, :]

        return vec_coord1, vec_coord2, coord_pxl
    
    def slice_axis(self, field, size_img=None, idx_axis=Z, coord_center=np.array([0, 0, 0])):
        '''
        Slice a field along a grid axis.
        
        Args
        field: field
        size_img: width of the image [cm]
        idx_axis: index of the slice axis
        coord_center: coordinate of the center of the image
        
        Returns
        img: image of the field in the slice plane
        '''
        field = self.get_field_input(field)
        if size_img == None: size_img = self.box_size
        
        idx_coord1, idx_coord2 = np.sort([(idx_axis+1)%3, (idx_axis+2)%3])
        idx_slice = np.searchsorted(self.coord1d[idx_axis], coord_center[idx_axis])
        
        idx_left = np.searchsorted(self.coord1d[idx_coord1], coord_center[idx_coord1] - size_img/2)
        idx_right = np.searchsorted(self.coord1d[idx_coord1], coord_center[idx_coord1] + size_img/2)
        idx_bottom = np.searchsorted(self.coord1d[idx_coord2], coord_center[idx_coord2] - size_img/2)
        idx_top = np.searchsorted(self.coord1d[idx_coord2], coord_center[idx_coord2] + size_img/2)
        
        img = field.take(indices=idx_slice, axis=idx_axis)[idx_left:idx_right, idx_bottom:idx_top]
        return img
    
    def slice_anyaxis(self, field, size_img=None, vec_camera=np.array([0, 0, 1]), vec_north=np.array([0, 1, 0]), coord_center=np.array([0, 0, 0]), num_pxl=128):
        '''
        Slice a field along any axis.
        
        Args
        field: field
        size_img (float): width of the image [cm]
        vec_camera: vector orieted perpendicular to the projection plane
        vec_north: vector oriented upwards in the image
        coord_center: coordinate of the center of the image plane
        num_pxl (int): width of the image in pixels
        
        Returns
        vec_coord1, vec_coord2: image plane vectors
        img: image of the field in the slice plane
        '''
        field = self.get_field_input(field)
        if size_img == None: size_img = self.box_size
        
        vec_coord1, vec_coord2, coord_pxl = self.getimgplane_anyaxis(size_img, vec_camera, vec_north=vec_north, coord_center=coord_center, num_pxl=num_pxl)
        idx_pxl = np.array([np.searchsorted(self.coord1d[i], coord_pxl[i]) for i in [X, Y, Z]])
        img = field[tuple(idx_pxl)]
        return vec_coord1, vec_coord2, img
    
    def proj_axis(self, field, size_img=None, size_sample=None, idx_axis=Z, coord_center=np.array([0, 0, 0]), weight=None):
        '''
        Project a field along a grid axis.
        
        Args
        field: field
        size_img (float): width of the image [cm]
        size_sample (float): length to sample for the projection [cm]
        idx_axis: index of the slice axis
        coord_center: coordinate of the center of the image plane
        weight: weight
        
        Returns
        img: projected image of the field 
        '''
        field, weight = self.get_field_input(field), self.get_field_input(weight)
        if np.all(weight) == None: weight = np.ones_like(field)
        field_weighted = field * weight
        if size_img == None: size_img = self.box_size
        if size_sample == None: size_sample = self.box_size
        
        idx_coord1, idx_coord2 = np.sort([(idx_axis+1)%3, (idx_axis+2)%3])
        idx_left = np.searchsorted(self.coord1d[idx_coord1], coord_center[idx_coord1] - size_img/2)
        idx_right = np.searchsorted(self.coord1d[idx_coord1], coord_center[idx_coord1] + size_img/2)
        idx_bottom = np.searchsorted(self.coord1d[idx_coord2], coord_center[idx_coord2] - size_img/2)
        idx_top = np.searchsorted(self.coord1d[idx_coord2], coord_center[idx_coord2] + size_img/2)
        idx_back = np.searchsorted(self.coord1d[idx_axis], coord_center[idx_axis] - size_sample/2)
        idx_for = np.searchsorted(self.coord1d[idx_axis], coord_center[idx_axis] + size_sample/2)
        
        field = field.take(np.arange(idx_back, idx_for), axis=idx_axis)
        field_weighted = field_weighted.take(np.arange(idx_back, idx_for), axis=idx_axis)
        img = (np.sum(field_weighted, axis=idx_axis) / np.sum(weight, axis=idx_axis))[idx_left:idx_right, idx_bottom:idx_top]
        return img
        
    def proj_anyaxis(self, field, size_img=None, size_sample=None, vec_camera=np.array([0, 0, 1]), vec_north=np.array([0, 1, 0]), coord_center=np.array([0, 0, 0]), num_pxl=128, num_sample=10000, weight=None):
        '''
        Project a field along any axis.
        
        Args
        field: field
        size_img (float): width of the image [cm]
        size_sample (float): length to sample for the projection [cm]
        vec_camera: vector orieted perpendicular to the projection plane
        vec_north: vector oriented upwards in the image
        coord_center: coordinate of the center of the image plane
        num_pxl (int): width of the image in pixels
        num_sample (int): number of samples for projection
        weight: weight
        
        Returns
        vec_coord1, vec_coord2: image plane vectors
        img: projected image of the field 
        '''
        field, weight = self.get_field_input(field), self.get_field_input(weight)
        if np.all(weight) == None: weight = np.ones_like(field)
        field_weighted = field * weight
        if size_img == None: size_img = self.box_size
        if size_sample == None: size_sample = self.box_size
        
        arr_sample = np.linspace(-size_sample/2, size_sample/2, num_sample)
        vec_coord1, vec_coord2, coord_pxl = self.getimgplane_anyaxis(size_img=size_img, vec_camera=vec_camera, vec_north=vec_north, coord_center=coord_center, num_pxl=num_pxl)
        
        img = np.zeros((num_pxl, num_pxl))
        for i in range(num_pxl):
            for j in range(num_pxl):
                coord_sample = coord_pxl[:, i, j, None] + vec_camera[:, None] * arr_sample[None, :]
                idx_sample = np.array([np.searchsorted(self.coord1d[i], coord_sample[i]) for i in [X, Y, Z]])
                img[i, j] = np.sum(field_weighted[tuple(idx_sample)]) / np.sum(weight[tuple(idx_sample)])

        return vec_coord1, vec_coord2, img
    
    def plot_img(self, img, size_img=None, ax=None, vmin=None, vmax=None, cmap='jet', do_log=False, unit=const.kpc):
        '''
        Plot an image.
        
        Args
        img: image
        size_img (float): size of the image
        ax: plotting axis
        vmin (float): minimum value on color scale
        vmax (float): maximum value on color scale
        cmap (str): colormap
        do_log (bool): take the log of the image
        unit (float): axes units
        
        Returns
        im: image object
        '''
        if size_img == None: size_img = self.box_size
        if ax == None: ax = plt.gca()
        if do_log: 
            img = np.log10(img)
            if vmin != None: vmin = np.log10(vmin)
            if vmax != None: vmax = np.log10(vmax)
        im = ax.imshow(img, origin='lower', extent=[-size_img/unit/2, size_img/unit/2, -size_img/unit/2, size_img/unit/2], vmin=vmin, vmax=vmax, cmap=cmap)
        return im
    
    def add_dis_to_ax(self, ax, dis, size_img=None, unit=const.kpc, label_unit='kpc', color='white'):
        '''
        Add a distance annotation to the axis.

        Args
        ax: axis object
        dis: distance to annotate
        size_img: size of the image
        unit: unit of the image size
        label_unit: label of the unit
        color: color of the annotation
        '''
        if size_img == None: size_img = self.box_size
        ax.plot([-0.9*size_img/unit/2, (-0.9*size_img/2 + dis)/unit], [0.9*size_img/unit/2, 0.9*size_img/unit/2], color='white', lw=3)
        ax.annotate('%.3g %s' % (dis/unit, label_unit), ((-0.9*size_img + dis)/unit/2, 0.85*size_img/2/unit), color='white', horizontalalignment='center', verticalalignment='top', fontsize=14)

    def mean(self, field, weight=None):
        ''' Return the mean value of a field. '''
        if np.all(weight) == None: weight = np.ones_like(field)
        field, weight = self.get_field_input(field), self.get_field_input(weight)
        field_mean = np.sum(field * weight) / np.sum(weight)
        return field_mean

    def sum(self, field, weight=None):
        ''' Return the sum value of a field. '''
        if np.all(weight) == None: weight = np.ones_like(field)
        field, weight = self.get_field_input(field), self.get_field_input(weight)
        field_sum = np.sum(field * weight)
        return field_sum
    
    @cached_property
    def n_H(self):
        ''' Hydrogen number density. '''
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
    def c_s(self):
        ''' Sound speed '''
        return np.sqrt(self.pressure / self.density)
    
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
        ''' Ionization fraction (n_HII / n_H) estimated from the Saha equation. '''
        lamb_dB = np.sqrt(const.h**2 / (2 * np.pi * const.m_e * const.k_B * self.temp)) # thermal deBroglie wavelength
        g0, g1 = 2, 1 # statistical weights
        A = np.ones_like(self.temp)
        B = (2 / (self.n_H * lamb_dB**3)) * (g1 / g0) * np.exp(-const.energy_HII / (const.k_B * self.temp))
        C = -B
        ion_frac = solve_quadratic(A, B, C)[1]
        ion_frac[ion_frac > 1] = 1.
        return ion_frac
    
    @cached_property
    def eps_sf(self):
        ''' Star formation efficiency. '''
        return calc_eps_sf(self.density, self.energy_turb, self.temp, self.dx_local, self.b_turb)

    @cached_property
    def t_ff(self):
        ''' Freefall time. '''
        return np.sqrt(3 * np.pi / (32 * const.G * self.density))
   
    @cached_property
    def sfr_density(self):
        ''' Star formation rate density. '''
        return self.eps_sf * self.density / self.t_ff
    
    @cached_property
    def entropy(self):
        ''' Entropy. '''
        return self.pressure / self.density

