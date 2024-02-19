# import MPI
from mpi4py import MPI

# import libraries for I/O
import sys, os
import argparse
from astropy.io import ascii

# import libraries for computation
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf

# import custom functions
from functions import get_dump_list, move_to_sim_dir, get_info, aexp_to_proper_time, get_numline, calc_eps_sf2, Hist
from config import BIRTH, DEATH
import const

# parse command line arguments
parser = argparse.ArgumentParser(prog='stardata.py', description='Generate histograms of star data.')
parser.add_argument('round', type=int, help='simulation round')
parser.add_argument('name', type=str, help='simulation name')
parser.add_argument('outfile', type=str, help='output filename')
parser.add_argument('-b', '--bturb', type=float, default=1.0, help='turbulence forcing parameter')
parser.add_argument('-l', '--epssfloc', type=float, default=1.0, help='local star formation efficiency')
parser.add_argument('-a', '--aexpmax', type=float, default=0.1, help='maximum scale factor')
parser.add_argument('-n', '--ncpu', type=int, default=384, help='number of CPUs')
args = parser.parse_args()

# useful constants
boxlen = 100 * const.Mpc / const.h0 # box size
rho_crit_h0e1 = 1.88e-29 # critical density for h0 = 1

# list 1d histogram parameters
histparam_list = [
    Hist('time', 0, 550*const.Myr, name='starmass', weight='mass', nbin=10*550, do_log=False),
    Hist('time', 0, 550*const.Myr, name='starnum', nbin=10*550, do_log=False, idx_event=DEATH),
    Hist('eps_sf', 1e-4, 1e1, weight='mass'),
    Hist('eps_sf', 1e-4, 1e1, idx_event=DEATH),
    Hist('b_turb', 1/3, 1.0, weight='mass', do_log=False),
    Hist('b_turb', 1/3, 1.0, do_log=False, idx_event=DEATH),
    Hist('metallicity', 1e-3*const.Z_sol, 1e1*const.Z_sol, weight='mass'),
    Hist('metallicity', 1e-3*const.Z_sol, 1e1*const.Z_sol, idx_event=DEATH),
    Hist('energy_turb', 1e11, 1e16, weight='mass'),
    Hist('energy_turb', 1e11, 1e16, idx_event=DEATH),
    Hist('density', 1e-22, 1e-18, weight='mass'),
    Hist('density', 1e-27, 1e-18, idx_event=DEATH),
    Hist('temp', 1e1, 1e4, weight='mass'),
    Hist('temp', 1e1, 1e9, idx_event=DEATH),
    Hist(['density', 'temp'], [1e-22, 1e1], [1e-18, 1e4], dim=2, weight='mass'),
    Hist(['density', 'temp'], [1e-27, 1e1], [1e-18, 1e9], dim=2, idx_event=DEATH),
    Hist('alpha_vir', 1e0, 1e5, weight='mass'),
    Hist('alpha_vir', 1e0, 1e5, idx_event=DEATH),
    Hist('mach_turb', 1e-2, 1e3, weight='mass'),
    Hist('mach_turb', 1e-2, 1e3, idx_event=DEATH),
    Hist(['alpha_vir', 'mach_turb'], [1e0, 1e-1], [1e5, 1e3], dim=2, weight='mass'),
    Hist(['alpha_vir', 'mach_turb'], [1e0, 1e-1], [1e5, 1e3], dim=2, idx_event=DEATH),
    Hist('density_mff', 1e-20, 1e-10, weight='mass', nbin=128),
    Hist(['density', 'eps_sf'], [1e-22, 1e-4], [1e-18, 1e1], dim=2, weight='mass'),
    Hist(['time', 'eps_sf'], [0, 1e-4], [550*const.Myr, 1e1], dim=2, weight='mass', do_log=[False, True], nbin=[550, 512])
]
num_hist = len(histparam_list)

# functions for derived quantities
def erf_wrapper(s, sigma_s):
    return erf((sigma_s**2 - s) / (np.sqrt(2) * sigma_s))

def calc_hist_density_mff(stardata, histparam):
    
    bin_edge = 10**np.linspace(np.log10(histparam.vmin_list[0]), np.log10(histparam.vmax_list[0]), histparam.nbin_list[0]+1)
    hist = np.zeros(histparam.nbin_list[0])
    
    aexp = proper_time_to_aexp_interp(stardata['time'])
    dx = aexp * boxlen / 2**stardata['level']
    b_turb = stardata['b_turb'] if args.bturb == 0. else args.bturb
    c_s = np.sqrt(const.k_B * stardata['temp'] / const.m_p) # sound speed
    mach_turb = np.sqrt(2/3 * stardata['energy_turb']) / c_s # turbulent Mach number
    alpha_vir = 15 / np.pi * c_s**2 * (1 + mach_turb**2) / (const.G * stardata['density'] * dx**2) # virial parameter
    s_crit = np.log(alpha_vir * (1 + 2 * mach_turb**4 / (1 + mach_turb**2))) # lognormal critical density for star formation
    sigma_s = np.sqrt(np.log(1 + b_turb**2 * mach_turb**2)) # standard deviation of the lognormal subgrid density distribution
    
    for i in range(histparam.nbin_list[0]):
        s_bin_min = np.maximum(np.log(bin_edge[i]) - np.log(stardata['density']), s_crit)
        s_bin_max = np.maximum(np.log(bin_edge[i+1]) - np.log(stardata['density']), s_crit)
        hist_per_part = (erf_wrapper(s_bin_min, sigma_s) - erf_wrapper(s_bin_max, sigma_s)) / (1. + erf_wrapper(s_crit, sigma_s))
        hist_per_part = np.nan_to_num(hist_per_part)
        hist[i] = np.sum(stardata['mass'] * hist_per_part)
    return hist

# list of columns in star logs
col_name_list = np.array(["event", "id", "level", "mass", "x_star", "y_star", "z_star", "velx_star", "vely_star", "velz_star", "density", "velx", "vely", "velz", "temp", "metallicity", "energy_turb", "mask", "b_turb", "tag", "time"])
col_unit_list = np.array(["dimless", "dimless", "dimless", "mass", "length", "length", "length", "vel", "vel", "vel", "density", "vel", "vel", "vel", "dimless", "dimless", "spec_energy", "dimless", "dimless", "dimless", "dimless"])

# prepare for eps_sf calculation
aexp_list = np.linspace(1e-3, 0.1, 1000)
time_list = np.array([aexp_to_proper_time(aexp_list[i]) for i in range(len(aexp_list))])
proper_time_to_aexp_interp = interp1d(time_list, aexp_list, fill_value='extrapolate') # quickly interpolate the scale factor given a proper time
age_universe = aexp_to_proper_time(1.0)

def read_starcat_1cpu(idx_cpu):
    ''' Read star catalogs for 1 CPU. '''
    # create empty histogram lists
    hist_list_1cpu = [np.zeros(histparam.nbin_list) for histparam in histparam_list]

    # iterate over dumps
    dump_list = get_dump_list()
    for dump in dump_list:
        info = get_info(dump)
        
        filename = "output_%.5d/stars_%.5d.out%.5d" % (dump, dump, idx_cpu)
        if not os.path.isfile(filename): continue # skip if star log file does not exist
        if get_numline(filename) <= 1: continue # skip if star log file is empty
        
        # read the star log
        with open(filename, "r") as f:
            stardata_tab = ascii.read(filename, names=col_name_list, data_start=1)
        
        # preprocess star log data
        stardata = {}
        for col_name in col_name_list:
            if col_name != 'mask': # mask data is sometimes corrupted
                stardata[col_name] = np.array(stardata_tab[col_name].data, dtype=float)
        if stardata['id'].size == 0: continue # skip if star data is empty

        # convert to cgs units
        age_universe = aexp_to_proper_time(1.0, Omega_m0=info.Omega_m0, Omega_k0=info.Omega_k0, Omega_L0=info.Omega_L0, H0=info.H0)
        stardata['time'] = stardata['time'] / info.H0 + age_universe
        aexp = proper_time_to_aexp_interp(stardata['time'])
        density_unit = const.Omega_m0 * rho_crit_h0e1 * const.h0**2 / aexp**3
        time_unit = aexp**2 / const.H0
        length_unit = aexp * boxlen
        unit = dict(density=density_unit, time=time_unit, length=length_unit, mass=(density_unit * length_unit**3), vel=(length_unit / time_unit), spec_energy=(length_unit / time_unit)**2, dimless=np.ones_like(aexp))
        for col_name, col_unit in zip(col_name_list, col_unit_list):
            if col_name != 'mask': # mask data is sometimes corrupted
                stardata[col_name] *= unit[col_unit]

        # add derived quantities to stardata
        aexp = proper_time_to_aexp_interp(stardata['time'])
        dx = aexp * boxlen / 2**stardata['level']
        c_s = np.sqrt(2/3 * const.k_B * stardata['temp'] / const.m_p) # sound speed
        stardata['mach_turb'] = np.sqrt(2/3 * stardata['energy_turb']) / c_s # turbulent Mach number
        stardata['alpha_vir'] = 15 / np.pi * c_s**2 * (1 + stardata['mach_turb']**2) / (const.G * stardata['density'] * dx**2) # virial parameter
        b_turb = stardata['b_turb'] if args.bturb == 0. else args.bturb
        stardata['eps_sf'] = calc_eps_sf2(stardata['density'], stardata['energy_turb'], stardata['temp'], dx, b_turb=b_turb, eps_sf_loc=args.epssfloc)
        
        # calculate histograms
        for i, histparam in enumerate(histparam_list):
            if histparam.name == 'density_mff':
                hist_list_1cpu[i] += calc_hist_density_mff(stardata, histparam)
            else:
                hist_list_1cpu[i] += histparam.calc_hist(stardata)

        # break if beyond max scale factor
        if info.aexp > args.aexpmax: break

    return hist_list_1cpu

if __name__ == '__main__':
    
    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    sys.stdout.write("Process %d of %d running on %s.\n" % (rank, size, name))

    # move to simulation directory
    move_to_sim_dir(args.round, args.name, do_print=(rank == 0))
    if rank == 0: os.system('mkdir -p starcat')

    # distribute cpus across processes
    ncpu_per_process = int(np.ceil(args.ncpu / size))
    idx_cpu_min = min(rank * ncpu_per_process, args.ncpu)
    idx_cpu_max = min((rank + 1) * ncpu_per_process, args.ncpu)
    if rank == 0: sys.stdout.write("%d CPUs per process\n" % ncpu_per_process)
    
    # create empty histogram list
    hist_list_1p = [np.zeros(histparam.nbin_list) for histparam in histparam_list]
   
    # run cpus in parallel
    if rank == 0: sys.stdout.write('Reading star catalogs\n')
    for idx_cpu in range(idx_cpu_min, idx_cpu_max+1):
        hist_list_1cpu = read_starcat_1cpu(idx_cpu)
        for i in range(num_hist):
            hist_list_1p[i] += hist_list_1cpu[i]

    # gather results from all processes
    if rank == 0: sys.stdout.write('Gathering data from all processes\n')
    hist_list_1p_list = comm.gather(hist_list_1p, root=0)

    # concatenate the results on rank 0
    if rank == 0:
        
        # create empty histogram list
        hist_list = [np.zeros(histparam.nbin_list) for histparam in histparam_list]
        
        # concatenate star catalogs from all processes
        sys.stdout.write('Concatenating data from all processes\n')
        for hist_list_1p in hist_list_1p_list:
            for i in range(num_hist):
                hist_list[i] += hist_list_1p[i]

        # package histograms and bin centers in a dictionary
        hist_dict = {}
        for i, histparam in enumerate(histparam_list):
            hist_dict[histparam.histname] = hist_list[i]
            hist_dict[histparam.pdfname] = histparam.calc_pdf(hist_list[i])
            for i in range(histparam.dim):
                hist_dict[histparam.binname_list[i]] = histparam.bin_center_list[i]

        # save histogram
        sys.stdout.write('Saving star catalogs\n')
        np.savez(os.path.join('starcat', args.outfile), **hist_dict)
        sys.stdout.write('Done!\n')
