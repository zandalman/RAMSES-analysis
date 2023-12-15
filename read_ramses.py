from modules import *

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

def get_star_cat(cgs=False, log=False, a_exp_max=1.0, list_of_dump=None):
    ''' 
    Get clump catalog. 
    
    Args
    cgs (bool): use cgs units
    log (bool): log progress
    a_exp_max (float): maximum expansion factor
    list_of_dump: list of dumps to read
        
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
    # col_names = ["event", "id", "level", "mass", "x_star", "y_star", "z_star", "vel_x_star", "vel_y_star", "vel_z_star", "density", "vel_x", "vel_y", "vel_z", "temp", "metallicity", "energy_turb", "mask", "tag", "tau"]
    col_names = ["event", "id", "level", "mass", "x_star", "y_star", "z_star", "vel_x_star", "vel_y_star", "vel_z_star", "density", "vel_x", "vel_y", "vel_z", "temp", "metallicity", "energy_turb", "mask", "b_turb", "tag", "tau"]
    num_dump, id, level, mass, x, y, z, density, temperature, metallicity, energy_turb, b_turb, event, time = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    if list_of_dump == None: list_of_dump = get_list_of_dump()
    if log: print("Reading from %d dumps" % len(list_of_dump))
    for dump in list_of_dump:
        print("Reading dump %d..." % dump)
        info = get_info(dump)
        if cgs:
            # compute the age of the Universe
            integrand = lambda a: (info.Omega_m0 * a**(-1) + info.Omega_k0 + info.Omega_L0 * a**2)**(-1/2)
            age_universe = quad(integrand, 0, 1)[0] / info.H0
        for i in range(0, info.ncpu):
            filename = "output_%.5d/stars_%.5d.out%.5d" % (dump, dump, i+1)
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    num_lines = len(f.readlines())
                if num_lines > 1:
                    star = ascii.read(filename, names=col_names, data_start=1)
                    num_dump = np.concatenate((num_dump, np.full_like(np.array(star['level'].data), dump)))
                    event = np.concatenate ((event, np.array(star['event'].data)))
                    id = np.concatenate((id, np.array(star['id'].data)))
                    level = np.concatenate((level, np.array(star['level'].data)))
                    metallicity = np.concatenate((metallicity, np.array(star['metallicity'].data)))
                    temperature = np.concatenate((temperature, np.array(star['temp'].data)))
                    b_turb = np.concatenate((b_turb, np.array(star['b_turb'].data)))
                    if cgs:
                        mass = np.concatenate((mass, np.array(star['mass'].data) * info.mass_unit))
                        x = np.concatenate((x, np.array(star['x_star'].data) * info.length_unit))
                        y = np.concatenate((y, np.array(star['y_star'].data) * info.length_unit))
                        z = np.concatenate((z, np.array(star['z_star'].data) * info.length_unit))
                        density = np.concatenate((density, np.array(star['density'].data) * info.density_unit))
                        energy_turb = np.concatenate((energy_turb, np.array(star['energy_turb'].data) * info.vel_unit**2))
                        time = np.concatenate((time, np.array(star['tau'].data) / info.H0 + age_universe))
                    else:
                        mass = np.concatenate((mass, np.array(star['mass'].data)))
                        x = np.concatenate((x, np.array(star['x_star'].data)))
                        y = np.concatenate((y, np.array(star['y_star'].data)))
                        z = np.concatenate((z, np.array(star['z_star'].data)))
                        density = np.concatenate((density, np.array(star['density'].data)))
                        energy_turb = np.concatenate((energy_turb, np.array(star['energy_turb'].data)))
                        time = np.concatenate((time, np.array(star['tau'].data)))
        if info.a_exp > a_exp_max: break
    # starbirth_cat = SimpleNamespace(num_dump=num_dump[event==BIRTH], id=id[event==BIRTH], level=level[event==BIRTH], mass=mass[event==BIRTH], coord=np.array([x[event==BIRTH], y[event==BIRTH], z[event==BIRTH]]), density=density[event==BIRTH], temperature=temperature[event==BIRTH], metallicity=metallicity[event==BIRTH], energy_turb=energy_turb[event==BIRTH], time=time[event==BIRTH])
    # stardeath_cat = SimpleNamespace(num_dump=num_dump[event==DEATH], id=id[event==DEATH], level=level[event==DEATH], mass=mass[event==DEATH], coord=np.array([x[event==DEATH], y[event==DEATH], z[event==DEATH]]), density=density[event==DEATH], temperature=temperature[event==DEATH], metallicity=metallicity[event==DEATH], energy_turb=energy_turb[event==DEATH], time=time[event==DEATH])
    starbirth_cat = SimpleNamespace(num_dump=num_dump[event==BIRTH], id=id[event==BIRTH], level=level[event==BIRTH], mass=mass[event==BIRTH], coord=np.array([x[event==BIRTH], y[event==BIRTH], z[event==BIRTH]]), density=density[event==BIRTH], temperature=temperature[event==BIRTH], metallicity=metallicity[event==BIRTH], energy_turb=energy_turb[event==BIRTH], b_turb=b_turb[event==BIRTH], time=time[event==BIRTH])
    stardeath_cat = SimpleNamespace(num_dump=num_dump[event==DEATH], id=id[event==DEATH], level=level[event==DEATH], mass=mass[event==DEATH], coord=np.array([x[event==DEATH], y[event==DEATH], z[event==DEATH]]), density=density[event==DEATH], temperature=temperature[event==DEATH], metallicity=metallicity[event==DEATH], energy_turb=energy_turb[event==DEATH], b_turb=b_turb[event==DEATH], time=time[event==DEATH])
    return starbirth_cat, stardeath_cat

def get_cool_tab(dump=None, n3=100):
    '''
    Read the RAMSES cooling table.

    Args
    dump (int): dump number
    n3 (int): size of the metallicity array

    Returns
    cool_tab (SimpleNamespace)
        n_H: Hydrogen number density
        temp: temperature
        Z: metallicity
        cool: cooling rate 
        heat: heating rate
        metal: cooling rate per Z / Z_sol i.e. cool_total = cool + metal * Z / Z_sol
        n_spec: species number density
        xion: ionization fraction
        mu: mean molecular weight
    '''
    if dump == None: dump = get_list_of_dump()[-1]
    cooling_table_path = os.path.join("output_%.5d" % dump, "cooling_%.5d.out" % dump)
    # read the file
    with FortranFile(cooling_table_path, 'r') as f:
        n1, n2 = f.read_ints('i')
        log_n_H = f.read_reals('f8')
        log_temp = f.read_reals('f8')
        log_cool = f.read_reals('f8')
        log_heat = f.read_reals('f8')
        log_cool_com = f.read_reals('f8')
        log_heat_com = f.read_reals('f8')
        log_metal = f.read_reals('f8')
        log_cool_prime = f.read_reals('f8')
        log_heat_prime = f.read_reals('f8')
        log_cool_com_prime = f.read_reals('f8')
        log_heat_com_prime = f.read_reals('f8')
        log_metal_prime = f.read_reals('f8')
        mu = f.read_reals('f8')
        log_n_spec = f.read_reals('f8')
    # package the data
    n_H = 10**log_n_H
    temp = 10**log_temp
    Z = np.logspace(-6, 1, n3) * const.Z_sol
    cool = 10**log_cool.reshape(n2, n1)
    log_cool_prime = log_cool_prime.reshape(n2, n1)
    cool_com = 10**log_cool_com.reshape(n2, n1)
    log_cool_com_prime = log_cool_com_prime.reshape(n2, n1)
    heat = 10**log_heat.reshape(n2, n1)
    log_heat_prime = log_heat_prime.reshape(n2, n1)
    heat_com = 10**log_heat_com.reshape(n2, n1)
    log_heat_com_prime = log_heat_com_prime.reshape(n2, n1)
    metal = 10**log_metal.reshape(n2, n1)
    log_metal_prime = log_metal_prime.reshape(n2, n1)
    n_spec = 10**log_n_spec.reshape(6, n2, n1)
    xion = 10**np.array([np.log10(n_spec)[0, i, :] - log_n_H for i in range(n2)]).T
    mu = mu.reshape(n2, n1)
    cool_tab = SimpleNamespace(n_H=n_H, temp=temp, Z=Z, cool=cool, log_cool_prime=log_cool_prime, cool_com=cool_com, log_cool_com_prime=log_cool_com_prime, heat=heat, log_heat_prime=log_heat_prime, heat_com=heat_com, log_heat_com_prime=log_heat_com_prime, metal=metal, log_metal_prime=log_metal_prime, n_spec=n_spec, xion=xion, mu=mu)
    return cool_tab

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
    id_star = np.array([])
    tau_starbirth = np.array([])
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
