analysis_dir = "/home/za9132/analysis"
save_dir = analysis_dir + "/figures/current"
sim_base_dir = "/home/za9132/scratch/romain"

defaults = {
    "unit": 1,
    "slice": 2, 
    "project": False,
    "do_integrate": True,
    "weight": 1., 
    "cond": 1., 
    "avg": True, 
    "do_log": True, 
    "slice_coord": 0,
    "max_pixels": None,
    "nlevels": 200,
    "cmap": "jet",
    "cbar_label": 'field',
    "cbar_tick_increment": None, 
    "plot_star": False,
    "plot_dm": False, 
    "isocontours": None, 
    "isocontour_field": None, 
    "color_isocontour": 'black', 
    "color_star": 'black',
    "color_dm": 'black', 
    "cbar_orientation": 'vertical', 
    "width": None,
    "title": None
}

halo_poptx = [0.50685686, -0.085267235, 1.9921783, -7.7235762]
halo_popty = [0.5065317, -0.049495054, 1.6965165, -5.8186623]
halo_poptz = [0.49536868, 0.093198848, -1.6449332, 4.735689]
