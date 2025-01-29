# import MPI
from mpi4py import MPI

# import libraries for I/O
import os, sys, re
from scipy.io import FortranFile

# import libraries for computation
import numpy as np

# import libraries for plotting
import matplotlib.pyplot as plt
import ffmpeg

# import custom functions
from functions import move_to_sim_dir, aexp_to_proper_time
import const

# parse command line arguments
args = sys.argv
round = int(args[1]) # simulation round
sim_name = args[2] # simulation name

# simulation parameters
boxlen = 100 * const.Mpc / const.h0 # box size [Mpc/h0]

def make_frame(nmapfile):

    # read star data
    stardata = np.load('starcat/data.npz')
    time1d = stardata['time1d']
    SFR1d = stardata['starmass_hist1d']
    
    # create figure and axes
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=1e-3, wspace=1e-3)
        
    idx_proj = 1
    coord1_idx, coord2_idx = np.sort([(idx_proj+1)%3, (idx_proj+2)%3])
    
    for j in range(3): # loop over columns (variables)
        
        ax = axs[j]
        ax.set_xticks([])
        ax.set_yticks([])
        
        var = ['dm', 'dens', 'stars'][j]
        vmin = [-14, 0, -16][j]
        vmax = [-11, 7, -9][j]
        cmap = ['cividis', 'inferno', 'afmhot'][j]
        mapfile = 'movie%d/%s_%.5d.map'%(idx_proj+1, var, nmapfile)
        
        # read map file
        with FortranFile(mapfile, 'r') as f:
            aexp, dx, dy, dz = f.read_reals('f8')
            nx, ny = f.read_ints('i')
            data = f.read_reals('f4')
        data = np.array(data).reshape(ny, nx)
        time = aexp_to_proper_time(aexp)

        # calculate frame size
        size_img_coord1 = [dx, dy, dz][coord1_idx] * boxlen / const.kpc # frame size [kpc]
        size_img_coord2 = [dx, dy, dz][coord2_idx] * boxlen / const.kpc # frame size [kpc]
        extent = [-size_img_coord1/2, size_img_coord1/2, -size_img_coord2/2, size_img_coord2/2]

        # create image
        im = ax.imshow(np.log10(data+1e-30), cmap=cmap, interpolation='nearest', origin='lower', extent=extent, vmin=vmin, vmax=vmax)

        # add colorbar
        cax = ax.inset_axes([0, -0.1, 1, 0.1])
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.set_xticks(cax.get_xticks()[:-1])
        cbar.set_label([r'$\log \Sigma_{\rm DM}$ [${\rm g/cm^2}$]', r'$\log N_{\rm gas}$ [${\rm H/cm^2}$]', r'$\log \Sigma_*$ [${\rm g/cm^2}$]'][j], fontsize=14)
            
    axs[2].plot([10, 60], [60, 60], color='white', lw=4)
    axs[2].annotate(r'$50~{\rm ckpc}$', xy=(35, 56), verticalalignment='top', horizontalalignment='center', color='white', fontsize=16)
            
    # plot SFR
    axtop = axs[0].inset_axes([0, 1, 3, 2/3])
    axtop.set_xticks([])
    axtop.plot(time1d/const.Myr, SFR1d/(const.M_sol/const.yr))
    axtop.set_ylabel(r'SFR [${\rm M_\odot/{\rm yr}}$]', fontsize=14)
    axtop.axvline(x=time/const.Myr, color='red', lw=2)
    axtop.annotate(r'$z = %.2f$' % (1/aexp-1) + '\n' + r'$t = %.2f~{\rm Myr}$' % (time/const.Myr), fontsize=20, xy=(0.01, 0.96), xycoords='axes fraction', verticalalignment='top')
    axtop.set_xlim(195)
    
    return fig

if __name__ == '__main__':

    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # move to movie directory
    move_to_sim_dir(round, sim_name, do_print=(rank==0))

    # get mapfile list
    file_list = os.listdir('movie1')
    nmapfile_list = np.array([int(re.findall(r'\d+', filename)[0]) for filename in sorted(file_list) if filename.startswith('dens') and filename.endswith('.map')])
    nmapfile_list = nmapfile_list[nmapfile_list>1000]
    num_mapfile = len(nmapfile_list)

    # distribute mapfiles across processes
    nmapfile_per_process = int(np.ceil(num_mapfile / size))
    idx_mapfile_min = min(rank * nmapfile_per_process, num_mapfile-1)
    idx_mapfile_max = min((rank+1) * nmapfile_per_process, num_mapfile-1)
    sys.stdout.write("Process %d of %d running on %s assigned frames %d-%d.\n" % (rank, size, name, idx_mapfile_min, idx_mapfile_max))

    # create movie in parallel
    if rank == 0: sys.stdout.write('Reading map files\n')
    for idx_mapfile in range(idx_mapfile_min, idx_mapfile_max):
        nmapfile = nmapfile_list[idx_mapfile]
        
        # make frame
        fig = make_frame(nmapfile)

        # save figure
        plt.show()
        plt.savefig(os.path.join('movieframes', "img-%.5d.png" % idx_mapfile), bbox_inches="tight", dpi=300)
        plt.close(fig)

    # wait for all frames
    comm.Barrier()
