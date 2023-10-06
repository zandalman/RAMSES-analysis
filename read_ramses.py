from modules import *


def get_stdout(cmd):
    ''' Return the standard output of a command line directive '''
    stdout = subprocess.check_output(cmd, shell=True).decode()
    return stdout


class Halo(object):
    '''
    Object for storing data from halo files.
    
    Args
    stdout_split: standard output of a halo search command
    
    Attrs
    idx (int): halo index
    ncell (int): number of dark matter particles in the halo
    coord: coordinates of the halo
    rho_max (float): max density in the halo in code units
    mass (float): mass of the halo in code units
    is_cgs (bool): quantities in cgs units
    '''
    def __init__(self, stdout_split):
        
        self.idx = int(stdout_split[0])
        self.ncell = int(stdout_split[1])
        self.coord = np.array(stdout_split[2:5], dtype=float)
        self.rho_max = float(stdout_split[5])
        self.mass = float(stdout_split[6])
        self.is_cgs = False
        
    def convert_to_cgs(self, length_unit, density_unit):
        ''' Convert attributes to cgs units. '''
        if not self.is_cgs:
            mass_unit = density_unit * length_unit**3
            self.coord *= length_unit
            self.rho_max *= density_unit
            self.mass *= mass_unit
            self.is_cgs = True
        
        
class Clump(object):
    '''
    Object for storing data from clump files.
    
    Args
    stdout_split: standard output of a clump search command
    
    Attrs
    idx (int): clump index
    lev (int): number of merging iterations away from halo
    parent_idx (int): index of the clump one level below
    ncell (int): number of dark matter particles in the clump
    coord: coordinates of the clump
    rho_border (float): max density on the clump boundary
    rho_max (float): max density in the clump
    mass (float): mass of the clump
    is_cgs (bool): quantities in cgs units
    '''
    def __init__(self, stdout_split):
        
        stdout_split = stdout.split()
        
        self.idx = int(stdout_split[0])
        self.level = int(stdout_split[1])
        self.parent_idx = int(stdout_split[2])
        self.ncell = int(stdout_split[3])
        self.coord = np.array(stdout_split[4:7], dtype=float)
        self.rho_border = float(stdout_split[7])
        self.rho_max = float(stdout_split[8])
        self.mass = float(stdout_split[9])
        self.is_cgs = False
        
    def convert_to_cgs(self):
        ''' Convert attributes to cgs units. '''
        if not self.is_cgs:
            self.coord *= length_unit
            self.rho_border *= density_unit
            self.rho_max *= density_unit
            self.mass *= mass_unit
            self.is_cgs = True
        

def get_biggest_halos(num, use_density=True):
    '''
    Find the biggest halos in the simulation and return a list of Halo objects.
    
    Args
    num (int): number of halos to find
    use_density (bool): use halo density to define "biggest", rather than halo mass
    
    Returns
    list_of_halo: list of Halo objects
    '''
    biggest_halo_by_density_stdout = get_stdout("cat output*/halo* | sort -r -nk 2 | head -n 1").split()
    biggest_halo_by_density = Halo(biggest_halo_by_density_stdout)
    
    biggest_halo_by_mass_stdout = get_stdout("cat output*/halo* | sort -r -gk 7 | head -n 1").split()
    biggest_halo_by_mass = Halo(biggest_halo_by_mass_stdout)
    
    if use_density:
        list_of_stdout_split = np.array(get_stdout("cat output*/halo* | sort -r -nk 2 | head -n %d" % num).split()).reshape(-1, 7)
    else:
        list_of_stdout_split = np.array(get_stdout("cat output*/halo* | sort -r -gk 7 | head -n %d" % num).split()).reshape(-1, 7)
    
    list_of_halo = [Halo(stdout_split) for stdout_split in list_of_stdout_split]
        
    if biggest_halo_by_density.idx != biggest_halo_by_mass.idx:
        warnings.warn("Biggest density and biggest mass halo are not the same.")
    
    if num == 1:
        return list_of_halo[0]
    else:
        return list_of_halo
        
    return list_of_halo


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
    
        