# import MPI
from mpi4py import MPI

# import libraries for I/O
import os, sys, subprocess
from scipy.io import FortranFile

# import libraries for computation
import numpy as np

# import libraries for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
img_dir = '"%s-proj%s' % (var, idx_proj+1)
unit = const.kpc / const.h0
unit_latex = r'[comoving ${\rm kpc}/h_0$]'
framerate = 24

# plot settings
plot_settings = {
    "dens": dict(cmap='inferno', vmin=0, vmax=7, cbar_step=1, cbar_label=r'log Gas Density'),
    "dm": dict(cmap='cividis', vmin=-14, vmax=-11, cbar_step=1, cbar_label=r'log Dark Matter Density'),
    "stars": dict(cmap='afmhot', vmin=-16, vmax=-10, cbar_step=1, cbar_label=r'log Star Density')
}

def make_frame(data, extent, cmap=None, vmin=None, vmax=None, cbar_step=None, cbar_label=None):

        # create figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # create image
        im = ax.imshow(data, cmap=cmap, interpolation='nearest', origin='lower', extent=extent, vmin=vmin, vmax=vmax)

        # create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_ticks = np.arange(vmin, vmax + cbar_step/2, cbar_step)
        plt.colorbar(im, ax=ax, cax=cax, ticks=cbar_ticks, label=cbar_label)

        # compute time
        time = aexp_to_proper_time(aexp)

        # create axes labels and title
        ax.set_xlabel(r"$%s$" % ['x', 'y', 'z'][coord1_idx] + unit_latex, fontsize=16)
        ax.set_ylabel(r"$%s$" % ['x', 'y', 'z'][coord2_idx] + unit_latex, fontsize=16)
        ax.set_title("$t = %.4f~{\rm Myr}$" % (time / const.Myr), fontsize=16)

        return fig

if __name__ == '__main__':

    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    sys.stdout.write("Process %d of %d running on %s.\n" % (rank, size, name))

    # move to movie directory
    move_to_sim_dir(args.round, args.name, do_print=(rank == 0))
    os.chdir('movie%d' % (idx_proj+1))
    if rank == 0: os.system('mkdir -p %s' % img_dir)

    # get mapfile list
    mapfile_list = [filename for filename in os.listdir() if filename.startswith(var) and filename.endswith('.map')]
    num_mapfile = len(mapfile_list)

    # distribute mapfiles across processes
    nmapfile_per_process = int(np.ceil(num_mapfile / size))
    idx_mapfile_min = min(rank * nmapfile_per_process, num_mapfile)
    idx_mapfile_max = min((rank + 1) * nmapfile_per_process, num_mapfile)
    if rank == 0: sys.stdout.write("%d mapfiles per process\n" % nmapfile_per_process)

    # create movie in parallel
    if rank == 0: sys.stdout.write('Reading map files\n')
    for idx_mapfile in range(idx_mapfile_min, idx_mapfile_max+1):
        mapfile = mapfile_list[idx_mapfile]
        
        # read map file
        with FortranFile(mapfile, 'r') as f:
            aexp, dx, dy, dz = f.read_reals('f8')
            nx, ny = f.read_ints('i')
            data = f.read_reals('f4')

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
    
    if rank == 0:
        
        # make the movie
        sys.stdout.write('Creating movie\n')
        mapfile_name = os.path.join(img_dir, "/img-%05d.png")
        movie_name = os.path.join("..", "$var-proj%d.mp4" % (idx_proj+1))
        process = (
            ffmpeg
            .input(mapfile_name)
            .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') # required to prevent error for odd pxl img size
            .output(movie_name, vcodec='libx264', pix_fmt='yuv420p', r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # clean up
        os.system("rm -r %s" % img_dir)

        sys.stdout.write('Done\n')
