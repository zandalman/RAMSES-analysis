from modules import *

import yt
from yt.units import dimensions
from yt.visualization.fixed_resolution import FixedResolutionBuffer

# add fields to yt

def _vorticity_x_new(field, data):
    return data["gas", "velocity_z_gradient_y"] - data["gas", "velocity_y_gradient_z"]

def _vorticity_y_new(field, data):
    return data["gas", "velocity_x_gradient_z"] - data["gas", "velocity_z_gradient_x"]

def _vorticity_z_new(field, data):
    return data["gas", "velocity_y_gradient_x"] - data["gas", "velocity_x_gradient_y"]

def _vorticity_magnitude(field, data):
    return np.sqrt(data["gas", "vorticity_x_new"]**2 + data["gas", "vorticity_y_new"]**2 + data["gas", "vorticity_z_new"]**2)

def _velocity_divergence_new(field, data):
    return data["gas", "velocity_x_gradient_x"] + data["gas", "velocity_y_gradient_y"] + data["gas", "velocity_z_gradient_z"]

def _bturb(field, data):
    return (1./3. + 2./3. * (data["gas", "velocity_divergence_new"]**2 / (data["gas", "vorticity_magnitude"]**2 + data["gas", "velocity_divergence_new"]**2))**3)

fields_to_add = {
    "vorticity_x_new": {"function": _vorticity_x_new, "units": "1/s"},
    "vorticity_y_new": {"function": _vorticity_y_new, "units": "1/s"},
    "vorticity_z_new": {"function": _vorticity_z_new, "units": "1/s"},
    "vorticity_magnitude": {"function": _vorticity_magnitude, "units": "1/s"},
    "velocity_divergence_new": {"function": _velocity_divergence_new, "units": "1/s"},
    "bturb": {"function": _bturb, "units": ""}
}

class SimYT(object):
    
    def __init__(self, sim_round, sim_name, dump, coord_center):

        self.sim_round = sim_round
        self.sim_name = sim_name
        self.sim_latex = sim_name_to_latex[self.sim_name]
        self.sim_dir = move_to_sim_dir(self.sim_round, self.sim_name)
        self.save_dir = os.path.join(save_dir, "round%d" % self.sim_round, self.sim_name)
        self.dump = dump
        info = get_info(self.dump)
        for var_name in info.__dict__:
            setattr(self, var_name, info.__dict__[var_name])
        self.coord_center = coord_center
        self.coord_center_code = self.coord_center / self.length_unit
        self.frb = None
        self.box_size = None
        self.num = None
        self.idx_slice = None

        info_file = os.path.join("output_%.5d" % dump, "info_%.5d.txt" % dump)
        self.ds = yt.load(info_file)
        self.ds.add_gradient_fields(("gas", "velocity_x"))
        self.ds.add_gradient_fields(("gas", "velocity_y"))
        self.ds.add_gradient_fields(("gas", "velocity_z"))
        for field_name, field_data in fields_to_add.items():
            self._add_field(field_name, field_data["function"], field_data["units"])
        self.ad = self.ds.all_data()
    
    def _add_field(self, name, func, units):

        self.ds.add_field(
            name=("gas", name),
            function=func,
            sampling_type="local",
            units=units,
            force_override=True
        )
    
    def _frb(self, box_size, idx_slice=Z, num=512):

        idx_coord1, idx_coord2 = np.sort([(idx_slice + 1) % 3, (idx_slice + 2) % 3])
        left_edge_code = (self.coord_center - box_size / 2) / self.length_unit
        right_edge_code = (self.coord_center + box_size / 2) / self.length_unit
        sl = self.ds.slice(idx_slice, self.coord_center_code[idx_slice])
        bounds = (left_edge_code[idx_coord1], right_edge_code[idx_coord1], left_edge_code[idx_coord2], right_edge_code[idx_coord2])
        self.frb = FixedResolutionBuffer(sl, bounds, (num, num))
        self.box_size, self.num, self.idx_slice = box_size, num, idx_slice
    
    def plot_slice(self, field_name, box_size, ax=None, idx_slice=Z, vmin=None, vmax=None, cmap='jet', label=None, do_log=False, num=512):

        if ax == None: ax = plt.gca()
        if (box_size != self.box_size) or (num != self.num) or (idx_slice != self.idx_slice):
            self._frb(box_size, idx_slice=idx_slice, num=num)
        field = self.frb[field_name].T
        if do_log: 
            field = np.log10(field)
            if vmin: vmin = np.log10(vmin)
            if vmax: vmax = np.log10(vmax)
        im = ax.imshow(field, cmap=cmap, extent=[-box_size/2/const.kpc, box_size/2/const.kpc, -box_size/2/const.kpc, box_size/2/const.kpc], origin='lower', vmin=vmin, vmax=vmax)
        label_coord_list = [r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]"]
        idx_coord1, idx_coord2 = np.sort([(idx_slice + 1) % 3, (idx_slice + 2) % 3])
        ax.set_xlabel(label_coord_list[idx_coord1])
        ax.set_ylabel(label_coord_list[idx_coord2])
        cbar = plt.colorbar(im, ax=ax)
        if label != None: cbar.set_label(label)
        ax.set_title(r'$a = %.3g$' % self.a_exp)

