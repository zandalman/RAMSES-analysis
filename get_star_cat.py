import os, sys
import numpy as np
from scipy.io import FortranFile
from read_ramses import *

# read command line arguments
args = sys.argv
sim_round = int(args[1]) # simulation round
sim_name = args[2] # simulation name
if len(args) > 3:
    aexp_max = float(args[3])
else:
    aexp_max = 1.0

# get starbirth and stardeath catalog
print("Reading star formation log files...")
move_to_sim_dir(sim_round, sim_name, do_print=True)
starbirth_cat, stardeath_cat = get_star_cat(cgs=True, log=True, aexp_max=aexp_max)

# save catalogs
print("Saving catalogs...")
np.savez("starbirth2", **starbirth_cat.__dict__)
np.savez("stardeath2", **stardeath_cat.__dict__)
print("Done!")
