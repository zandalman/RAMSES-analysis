# import MPI
from mpi4py import MPI

# import libraries for I/O
import sys, os
import argparse
from astropy.io import ascii

# import libraries for computation
import numpy as np
from scipy.interpolate import interp1d

# import custom functions
from functions import get_dump_list, move_to_sim_dir, get_info, aexp_to_proper_time, get_numline
from config import BIRTH, DEATH
import const

# parse command line arguments
parser = argparse.ArgumentParser(prog='stardata.py', description='Generate histograms of star data.')
parser.add_argument('round', type=int, help='simulation round')
parser.add_argument('name', type=str, help='simulation name')
parser.add_argument('outfile', type=str, help='output filename')
parser.add_argument('-a', '--aexpmax', type=float, default=0.1, help='maximum scale factor')
parser.add_argument('-n', '--ncpu', type=int, default=384, help='number of CPUs')
args = parser.parse_args()

# useful constants
boxlen = 100 * const.Mpc / const.h0 # box size
rho_crit_h0e1 = 1.88e-29 # critical density for h0 = 1

# list of columns in star logs
col_name_list = np.array(["event", "id", "level", "mass", "x_star", "y_star", "z_star", "velx_star", "vely_star", "velz_star", "density", "velx", "vely", "velz", "temp", "metallicity", "energy_turb", "mask", "b_turb", "tag", "time"])
col_unit_list = np.array(["dimless", "dimless", "dimless", "mass", "length", "length", "length", "vel", "vel", "vel", "density", "vel", "vel", "vel", "dimless", "dimless", "spec_energy", "dimless", "dimless", "dimless", "dimless"])
col_name_select_list = np.array(["id", "level", "mass", "x_star", "y_star", "z_star", "velx_star", "vely_star", "velz_star", "density", "velx", "vely", "velz", "temp", "metallicity", "energy_turb", "b_turb", "time"])
ncol_select = len(col_name_select_list)

# prepare for eps_sf calculation
aexp_list = np.linspace(1e-3, 0.1, 1000)
time_list = np.array([aexp_to_proper_time(aexp_list[i]) for i in range(len(aexp_list))])
proper_time_to_aexp_interp = interp1d(time_list, aexp_list, fill_value='extrapolate') # quickly interpolate the scale factor given a proper time
age_universe = aexp_to_proper_time(1.0)

def read_starcat_1cpu(idx_cpu, id_list):
    ''' Read star catalogs for 1 CPU. '''
    # create empty histogram lists
    data_1cpu = np.zeros((id_list.size, ncol_select))

    # iterate over dumps
    dump_list = get_dump_list()
    for dump in dump_list:
        info = get_info(dump)
        
        filename = "output_%.5d/stars_%.5d.out%.5d" % (dump, dump, idx_cpu)
        if not os.path.isfile(filename): continue # skip if star log file does not exist
        if get_numline(filename) <= 1: continue # skip if star log file is empty
        
        # read the star log
        stardata_tab = ascii.read(filename, names=col_name_list, data_start=1)
        
        # preprocess star log data
        stardata = {}
        for col_name in col_name_list:
            if col_name != 'mask': # mask data is sometimes corrupted
                stardata[col_name] = np.array(stardata_tab[col_name].data, dtype=float)
        if stardata['density'].size == 0: continue # skip if star data is empty
        if np.sum(stardata['event']==BIRTH) == 0: continue # skip if no birth events
        if stardata['density'].size != stardata['event'].size: continue # skip if columns are inconsistent (not sure why)

        # convert to cgs units
        age_universe = aexp_to_proper_time(1.0, Omega_m0=info.Omega_m0, Omega_k0=info.Omega_k0, Omega_L0=info.Omega_L0, H0=info.H0)
        stardata['time'] = stardata['time'] / info.H0 + age_universe
        aexp = proper_time_to_aexp_interp(stardata['time'])
        density_unit = const.Omega_m0 * rho_crit_h0e1 * const.h0**2 / aexp**3
        time_unit = aexp**2 / const.H0
        length_unit = aexp * boxlen
        unit = dict(density=density_unit, time=time_unit, length=length_unit, mass=(density_unit * length_unit**3), vel=(length_unit / time_unit), spec_energy=(length_unit / time_unit)**2, dimless=np.ones_like(aexp))
        for col_name, col_unit in zip(col_name_list, col_unit_list):
            if col_name not in ['event', 'mask']:
                stardata[col_name] = (stardata[col_name] * unit[col_unit])[stardata['event']==BIRTH]

        # search star data
        is_match = np.isin(stardata['id'], id_list, assume_unique=True)
        if np.sum(is_match) == 0: continue

        # find matches
        sorter = np.argsort(id_list)
        idx_match = sorter[np.searchsorted(id_list, stardata['id'][is_match], sorter=sorter)]
        
        for i, col_name in enumerate(col_name_select_list):
            data_1cpu[idx_match, i] = stardata[col_name][is_match]

        # break if beyond max scale factor
        if info.aexp > args.aexpmax: break

    return data_1cpu

if __name__ == '__main__':
    
    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # move to simulation directory
    move_to_sim_dir(args.round, args.name, do_print=(rank == 0))
    if rank == 0: os.system('mkdir -p starcat')

    # read ids
    id_list = np.load('starcat/ids.npz')['id']
    data_1p = np.zeros((id_list.size, ncol_select))

    # distribute cpus across processes
    ncpu_per_process = int(np.ceil(args.ncpu / size))
    idx_cpu_min = min(rank * ncpu_per_process, args.ncpu-1)
    idx_cpu_max = min((rank+1) * ncpu_per_process, args.ncpu-1)
    sys.stdout.write("Process %d of %d running on %s assigned CPUs %d-%d\n" % (rank, size, name, idx_cpu_min, idx_cpu_max))
   
    # run cpus in parallel
    if rank == 0: sys.stdout.write('Reading star catalogs\n')
    for idx_cpu in range(idx_cpu_min, idx_cpu_max+1):
        data_1p += read_starcat_1cpu(idx_cpu, id_list)

    # gather results from all processes
    if rank == 0: sys.stdout.write('Gathering data from all processes\n')
    data_list = np.array([comm.gather(data_1p, root=0)])

    # concatenate the results on rank 0
    if rank == 0:

        data_dict = {}
        for i, col_name in enumerate(col_name_select_list):
            data_dict[col_name] = np.sum(data_list[:,:,i], axis=0)

        # save histogram
        sys.stdout.write('Saving star catalogs\n')
        np.savez(os.path.join('starcat', args.outfile), **data_dict)
        sys.stdout.write('Done!\n')
