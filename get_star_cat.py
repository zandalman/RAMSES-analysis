import os, sys
import numpy as np
from scipy.io import FortranFile
from read_ramses import *

# read command line arguments
args = sys.argv
sim_round = int(args[1]) # simulation round
sim_name = args[2] # simulation name
if len(args) > 3:
    a_exp_max = float(args[3])
else:
    a_exp_max = 1.0

# get starbirth and stardeath catalog
print("Reading star formation log files...")
move_to_sim_dir(sim_round, sim_name, do_print=True)
starbirth_cat, stardeath_cat = get_star_cat(cgs=True, log=True, a_exp_max=a_exp_max)

'''
# read particle files
print("Reading particle files...")
dump = get_dump(a_exp=a_exp_max)
id_star, time_starbirth = read_partfile(dump)

# add birthtime to starbirth catalog by mathcing ids with particle file data
print("Calculating birthtimes...")
starbirth_cat.time_starbirth = np.zeros_like(starbirth_cat.mass, dtype=float)
starbirth_cat.flag_unknowntime = np.zeros_like(starbirth_cat.mass, dtype=int)

for i, id_star_cat in enumerate(starbirth_cat.id.astype(int)):
    time_starbirth_match = time_starbirth[id_star == id_star_cat]
    if len(time_starbirth_match) > 0: 
        starbirth_cat.time_starbirth[i] = time_starbirth[0]
    else:
        starbirth_cat.flag_unknowntime[i] = 1
'''

# save catalogs
print("Saving catalogs...")
np.savez("starbirth", **starbirth_cat.__dict__)
np.savez("stardeath", **stardeath_cat.__dict__)
print("Done!")
