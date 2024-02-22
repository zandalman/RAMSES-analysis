# import MPI
from mpi4py import MPI

# import libraries for I/O
import os, sys
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
idx_proj = int(args[3])-1 # projection coordinate index
var = args[4] # variable ['dens', 'dm', 'stars']
coord1_idx, coord2_idx = np.sort([(idx_proj+1)%3, (idx_proj+2)%3])

# simulation parameters
boxlen = 100 * const.Mpc / const.h0 # box size [Mpc/h0]

# movie parameters
img_dir = '%s-proj%s' % (var, idx_proj+1)
unit = const.kpc / const.h0
unit_latex = r'[${\rm ckpc}/h_0$]'
framerate = 24

# plot settings
plot_settings = {
    "dens": dict(cmap='inferno', vmin=0, vmax=7, cbar_step=1, cbar_label=r'$\log \rho$ [${\rm H/cm^2}$]', do_log=True),
    "dm": dict(cmap='cividis', vmin=-14, vmax=-11, cbar_step=1, cbar_label=r'$\log \rho_{\rm dm}$ [${\rm g/cm^2}$]', do_log=True),
    "stars": dict(cmap='afmhot', vmin=-16, vmax=-9, cbar_step=1, cbar_label=r'$\log \rho_{\rm star}$ [${\rm g/cm^2}$]', do_log=True)
}

def make_frame(data, extent, cmap=None, vmin=None, vmax=None, cbar_step=None, cbar_label=None, do_log=None):

    # create figure and axes
    if do_log: data = np.log10(data + 1e-30)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_axes([0, 0, 1, 1])
    if do_SFR_plot: ax2 = fig.add_axes([0, 1, 1, 0.3])
    cax = fig.add_axes([1, 0, 0.05, 1])
    
    # create image
    im = ax1.imshow(data, cmap=cmap, interpolation='nearest', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    
    # create colorbar
    cbar_ticks = np.arange(vmin, vmax + cbar_step/2, cbar_step)
    cbar = plt.colorbar(im, cax=cax, ticks=cbar_ticks)
    cbar.set_label(label=cbar_label, fontsize=16)

    # compute time
    time = aexp_to_proper_time(aexp)
    
    # plot SFR
    if do_SFR_plot:
        ax2.plot(time1d, SFR1d/(const.M_sol/const.yr))
        ax2.axvline(x=time, lw=2, color='red')
        ax2.set_xticks([])
        plt.setp(ax2.get_xticklabels(), visible=False)

    # create axes labels and title
    ax1.set_xlabel(r"$%s$ " % ['x', 'y', 'z'][coord1_idx] + unit_latex, fontsize=16)
    ax1.set_ylabel(r"$%s$ " % ['x', 'y', 'z'][coord2_idx] + unit_latex, fontsize=16)
    if do_SFR_plot: 
        ax2.set_ylabel(r'SFR [$M_\odot/{\rm yr^{-1}}$]', fontsize=16)
        ax2.set_title(r"$t = %.4f~{\rm Myr}$" % (time / const.Myr), fontsize=16)
    else:
        ax1.set_title(r"$t = %.4f~{\rm Myr}$" % (time / const.Myr), fontsize=16)
    
    return fig

if __name__ == '__main__':

    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # move to movie directory
    move_to_sim_dir(round, sim_name, do_print=(rank==0))
    if os.path.isfile('starcat/data.npz'):
        stardata = np.load('starcat/data.npz')
        time1d = stardata['time1d']
        SFR1d = stardata['starmass_hist1d'] / np.diff(time1d)[0]
        do_SFR_plot = True
    else:
        do_SFR_plot = False
    os.chdir('movie%d' % (idx_proj+1))

    # get mapfile list
    mapfile_list = [filename for filename in sorted(os.listdir()) if filename.startswith(var) and filename.endswith('.map')]
    num_mapfile = len(mapfile_list)

    # distribute mapfiles across processes
    nmapfile_per_process = int(np.ceil(num_mapfile / size))
    idx_mapfile_min = min(rank * nmapfile_per_process, num_mapfile-1)
    idx_mapfile_max = min((rank+1) * nmapfile_per_process, num_mapfile-1)
    sys.stdout.write("Process %d of %d running on %s assigned frames %d-%d.\n" % (rank, size, name, idx_mapfile_min, idx_mapfile_max))

    # create movie in parallel
    if rank == 0: sys.stdout.write('Reading map files\n')
    for idx_mapfile in range(idx_mapfile_min, idx_mapfile_max):
        mapfile = mapfile_list[idx_mapfile]
        
        # read map file
        with FortranFile(mapfile, 'r') as f:
            aexp, dx, dy, dz = f.read_reals('f8')
            nx, ny = f.read_ints('i')
            data = f.read_reals('f4')
        data = np.array(data).reshape(ny, nx)

        # calculate frame size
        size_img_coord1 = [dx, dy, dz][coord1_idx] * boxlen / unit # frame size [kpc/h0]
        size_img_coord2 = [dx, dy, dz][coord2_idx] * boxlen / unit # frame size [kpc/h0]
        extent = [-size_img_coord1/2, size_img_coord1/2, -size_img_coord2/2, size_img_coord2/2]
        
        # make frame
        fig = make_frame(data, extent, **plot_settings[var])

        # save figure
        plt.show()
        plt.savefig(os.path.join(img_dir, "img-%.5d.png" % idx_mapfile), bbox_inches="tight", dpi=300)
        plt.close(fig)

    # wait for all frames
    comm.Barrier()
