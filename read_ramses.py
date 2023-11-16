from modules import *

X, Y, Z = 0, 1, 2 # cartesian coordinate indices
S, PH2, Z = 0, 1, 2 # cylindrical coordinate indices
R, H, PH = 0, 1, 2 # spherical coordinate indices
CART, CYL, SPH = 0, 1, 2 # coordinate systems
HYDRO, DM, STAR = 0, 1, 2 # object categories
BIRTH, DEATH = 0, 1 # star log categories
epsilon = 1e-30 # small number
class Terminal:
    ''' Context manager for running commands in the terminal from Python. '''
    def __enter__(self):
        self.cwd = os.getcwd()
        return None
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self.cwd)


def get_stdout(cmd):
    ''' Return the standard output of a command line directive '''
    stdout = subprocess.check_output(cmd, shell=True).decode()
    return stdout


def get_dump(a_exp):
    ''' Get the dump number for a given expansion factor. '''
    stdout_split = get_stdout("grep aexp output_00*/info*").split()
    list_of_dump = np.array([int(s[7:12]) for s in stdout_split[::3]])
    list_of_a_exp = np.array([float(s) for s in stdout_split[2::3]])
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


def get_halo_cat(dump, cgs=False):
    ''' 
    Get halo catalog. 
    
    Args
    dump (int): dump number
    cgs (bool): use cgs units
    
    Returns
    halo_cat: halo catalog SimpleNamespace
        idx (int): halo index
        ncell (int): number of dark matter particles in the halo
        coord: coordinates of the halo
        density_max (float): max density in the halo in code units
        mass (float): mass of the halo in code units
    '''
    info = get_info(dump)
    idx, ncell, x, y, z, density_max, mass = [], [], [], [], [], [], []
    for i in range(0, info.ncpu):
        filename = "output_%.5d/halo_%.5d.txt%.5d" % (dump, dump, i+1)
        if os.path.isfile(filename):
            halo = ascii.read(filename)
            idx += list(halo['index'].data)
            ncell += list(halo['ncell'].data)
            x += list(halo['peak_x'].data)
            y += list(halo['peak_y'].data)
            z += list(halo['peak_z'].data)
            density_max += list(halo['rho+'].data)
            mass += list(halo['mass'].data)
    halo_cat = SimpleNamespace(idx=np.array(idx), ncell=np.array(ncell), coord=np.array([x, y, z]), density_max=np.array(density_max), mass=np.array(mass))
    if cgs: 
        halo_cat.coord *= info.length_unit
        halo_cat.density_max *= info.density_unit
        halo_cat.mass *= info.mass_unit
    return halo_cat

def get_clump_cat(dump, cgs=False):
    ''' 
    Get clump catalog. 
    
    Args
    dump (int): dump number
    cgs (bool): use cgs units
        
    Returns
    clump_cat: halo catalog SimpleNamespace
        idx (int): clump index
        level (int): number of merging iterations away from halo
        parent_idx (int): index of the clump one level below
        ncell (int): number of dark matter particles in the clump
        coord: coordinates of the clump
        rho_border (float): max density on the clump boundary
        rho_max (float): max density in the clump
        rho_avg (float): average density in the clump
        mass (float): mass of the clump
    '''
    info = get_info(dump)
    idx, level, parent_idx, ncell, x, y, z, density_border, density_max, density_avg, mass = [], [], [], [], [], [], [], [], [], [], []
    for i in range(0, info.ncpu):
        filename = "output_%.5d/clump_%.5d.txt%.5d" % (dump, dump, i+1)
        if os.path.isfile(filename):
            clump = ascii.read(filename)
            idx += list(clump['index'].data)
            level += list(clump['lev'].data)
            parent_idx += list(clump['parent'].data)
            ncell += list(clump['ncell'].data)
            x += list(clump['peak_x'].data)
            y += list(clump['peak_y'].data)
            z += list(clump['peak_z'].data)
            density_border += list(clump['rho-'].data)
            density_max += list(clump['rho+'].data)
            density_avg += list(clump['rho_av'].data)
            mass += list(clump['mass_cl'].data)
    clump_cat = SimpleNamespace(idx=np.array(idx), level=np.array(level), parent_idx=np.array(parent_idx), ncell=np.array(ncell), coord=np.array([x, y, z]), density_border=np.array(density_border), density_max=np.array(density_max), density_avg=np.array(density_avg), mass=np.array(mass))
    if cgs:
        clump_cat.coord *= info.length_unit
        clump_cat.density_border *= info.density_unit
        clump_cat.density_max *= info.density_unit
        clump_cat.density_avg *= info.density_unit
        clump_cat.mass *= info.mass_unit
    return clump_cat


def get_star_cat(cgs=False, log=False, a_exp_max=1.0):
    ''' 
    Get clump catalog. 
    
    Args
    cgs (bool): use cgs units
    log (bool): log progress
    max_a_exp (float): maximum expansion factor
        
    Returns
    starbirth_cat, stardeath_cat: star catalog SimpleNamespace for star formation and supernovae events
        num_dump (int): dump in which event occurs
        level (int): AMR level of cell
        mass (float): mass of star particle
        coord (float): coordinate of star particle
        density (float): density of cell
        pressure (float): pressure of cell
        metallicity (float): metallicity of cell
        energy_turb (float): turbulent energy of cell
    '''
    num_dump, id, level, mass, x, y, z, density, temperature, metallicity, energy_turb, event = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    list_of_dump = get_list_of_dump()
    if log: print("Reading from %d dumps" % len(list_of_dump))
    for dump in list_of_dump:
        print("Reading dump %d..." % dump)
        info = get_info(dump)
        for i in range(0, info.ncpu):
            filename = "output_%.5d/stars_%.5d.out%.5d" % (dump, dump, i+1)
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    num_lines = len(f.readlines())
                if num_lines > 1:
                    star = ascii.read(filename)
                    num_dump = np.concatenate((num_dump, np.full_like(np.array(star['ilevel'].data), dump)))
                    id = np.concatenate((id, np.array(star['id'].data)))
                    level = np.concatenate((level, np.array(star['ilevel'].data)))
                    metallicity = np.concatenate((metallicity, np.array(star['u6'].data)))
                    event = np.concatenate ((event, np.array(star['event'].data)))
                    if cgs:
                        mass = np.concatenate((mass, np.array(star['mp'].data) * info.mass_unit))
                        x = np.concatenate((x, np.array(star['xp1'].data) * info.length_unit))
                        y = np.concatenate((y, np.array(star['xp2'].data) * info.length_unit))
                        z = np.concatenate((z, np.array(star['xp3'].data) * info.length_unit))
                        density = np.concatenate((density, np.array(star['u1'].data) * info.density_unit))
                        temperature = np.concatenate((temperature, np.array(star['u5'].data)))
                        energy_turb = np.concatenate((energy_turb, np.array(star['u7'].data) * info.vel_unit**2))
                    else:
                        mass = np.concatenate((mass, np.array(star['mp'].data)))
                        x = np.concatenate((x, np.array(star['xp1'].data)))
                        y = np.concatenate((y, np.array(star['xp2'].data)))
                        z = np.concatenate((z, np.array(star['xp3'].data)))
                        density = np.concatenate((density, np.array(star['u1'].data)))
                        temperature = np.concatenate((temperature, np.array(star['u5'].data)))
                        energy_turb = np.concatenate((energy_turb, np.array(star['u7'].data)))
        if info.a_exp > a_exp_max: break
    starbirth_cat = SimpleNamespace(num_dump=num_dump[event==BIRTH], id=id[event==BIRTH], level=level[event==BIRTH], mass=mass[event==BIRTH], coord=np.array([x[event==BIRTH], y[event==BIRTH], z[event==BIRTH]]), density=density[event==BIRTH], pressure=pressure[event==BIRTH], metallicity=metallicity[event==BIRTH], energy_turb=energy_turb[event==BIRTH])
    stardeath_cat = SimpleNamespace(num_dump=num_dump[event==DEATH], id=id[event==DEATH], level=level[event==DEATH], mass=mass[event==DEATH], coord=np.array([x[event==DEATH], y[event==DEATH], z[event==DEATH]]), density=density[event==DEATH], pressure=pressure[event==DEATH], metallicity=metallicity[event==DEATH], energy_turb=energy_turb[event==DEATH])
    return starbirth_cat, stardeath_cat


def read_partfile(dump):
    '''
    Read the particle files and extract the star ids and birthtimes.

    Args
    dump (int): dump number

    Returns
    id_star: array of star particle ids
    time_starbirth: array of star birthtimes
    '''
    info = get_info(dump)
    
    # initialize arrays
    id_star = np.array([])
    tau_starbirth = np.array([])

    # read particle files
    for i in range(1, info.ncpu+1):
        partfile = os.path.join("output_%.5d" % dump, "part_%.5d.out%.5d" % (dump, i))
        with FortranFile(partfile, 'r') as f:
            _, _, _ = f.read_ints('i'), f.read_ints('i'), f.read_ints('i')
            _, _, _, _, _ = f.read_reals('f8'), f.read_reals('f4'), f.read_reals('f8'), f.read_reals('f8'), f.read_reals('f4')
            _, _, _ = f.read_reals('f8'), f.read_reals('f8'), f.read_reals('f8') # position
            _, _, _ = f.read_reals('f8'), f.read_reals('f8'), f.read_reals('f8') # velocity
            _ = f.read_reals('f8') # mass
            id_part = f.read_ints('i') # id
            _ = f.read_ints('i') # level
            type_part = f.read_ints('b') # family
            _ = f.read_ints('b') # tag
            tau_partbirth = f.read_reals('f8') # birthtime
            _ = f.read_reals('f8') # metallicity

            id_star = np.concatenate((id_star, id_part[type_part==STAR]))
            tau_starbirth = np.concatenate((tau_starbirth, tau_partbirth[type_part==STAR]))

    integrand = lambda a: (info.Omega_m0 * a**(-1) + info.Omega_k0 + info.Omega_L0 * a**2)**(-1/2)
    age_universe = quad(integrand, 0, 1)[0] / info.H0
    time_starbirth = tau_starbirth / info.H0 + age_universe

    return id_star, time_starbirth


def safe_savez(filename, **kwargs):
    ''' Save data to an npz file, but confirm before overwriting existing files. '''
    if filename in os.listdir() or filename + ".npz" in os.listdir():
        do_overwrite = input("File '%s' already exists. Would you like to overwrite? [yes/no]: " % filename) == "yes"
        if do_overwrite: 
            print("Overwriting file.")
            np.savez(filename, **kwargs)
        else:
            print("Data not saved.")
    else:
        np.savez(filename, **kwargs)


def move_to_sim_dir(sim_round, sim_name, do_print=True):
    ''' Move to simulation directory. '''
    sim_dir = os.path.join(config.sim_base_dir, "round%d" % sim_round, sim_name)
    os.chdir(sim_dir)
    if do_print: print("Moving to directory '%s'." % sim_dir)
    return sim_dir


def read_logfile():
    ''' 
    Read the simulation log files.

    Returns
    time: running time
    a_exp: expansion factor
    num_oct_level: number of octs per level
    num_oct_total: number of octs total
    '''
    dump = get_list_of_dump()[-1]
    info = get_info(dump)
    amr_level_max_coarse = info.amr_level_max + info.amr_level_coarse
    
    time, a_exp = [], []
    num_oct_level = [[] for i in range(amr_level_max_coarse)]

    list_of_logfile = sorted(get_stdout("ls *.log").split())
    time_now = 0

    for logfile in list_of_logfile:
        
        stdout_split = get_stdout("grep -B1 -n 'Computing new cooling table' %s | awk -F ' ' '{print $9}'" % logfile).split()
        for s in stdout_split:
            try: a_exp.append(float(s))
            except: a_exp.append(a_exp[-1])
        
        stdout_split = get_stdout("grep -A%d -n 'Mesh structure' %s | awk -F ' ' '{print $5}'" % (amr_level_max_coarse, logfile)).split()
        for i in range(amr_level_max_coarse):
            for s in stdout_split[i::amr_level_max_coarse]:
                try: num_oct_level[i].append(int(s))
                except: num_oct_level[i].append(0)

        stdout_split = get_stdout("grep -n 'Total running time' %s | awk -F ' ' '{print $5}'" % logfile).split()
        time += [float(t) + time_now for t in stdout_split]
        time_now = time[-1]

    a_exp = np.array(a_exp)
    num_oct_level = np.array(num_oct_level)
    num_oct_tot = np.sum(num_oct_level, axis=0)
    time = np.array(time)

    return time, a_exp, num_oct_level, num_oct_tot

        
def read_timer():
    '''  
    Read the timer file. 
    '''
    list_of_timing_category = ["coarse levels", "refine", "load balance", "particles", "rho", "poisson", "feedback", "cooling", "hydro - set unew", "hydro - godunov", "hydro - rev ghostzones", "hydro - set uold", "hydro - ghostzones", "hydro upload fine", "flag", "io", "courant", "movie"]
    list_of_stat = ["min", "max", "avg", "std", "relstd", "percent"]

    timing = {category: {stat: [] for stat in list_of_stat} for category in list_of_timing_category}
    timing["dump"] = []

    for file in sorted(os.listdir()):
        if file[:6] == 'output':
            timer_path = os.path.join(file, "timer%s.txt" % file[6:])
            with open(timer_path, "r") as f:
                timing["dump"].append(int(file[7:]))
                list_of_category_copy = list_of_timing_category.copy()
                for line in f.readlines()[3:-1]:
                    line_split = line.split()
                    category = " ".join(line_split[8:])
                    timing[category]["min"].append(float(line_split[0]))
                    timing[category]["max"].append(float(line_split[2]))
                    timing[category]["avg"].append(float(line_split[1]))
                    timing[category]["std"].append(float(line_split[3]))
                    timing[category]["relstd"].append(float(line_split[4]))
                    timing[category]["percent"].append(float(line_split[5]))
                    list_of_category_copy.remove(category)
                for category in list_of_category_copy:
                    for stat in list_of_stat:
                        timing[category][stat].append(0.)
    return timing
