X, Y, Z = 0, 1, 2 # cartesian coordinate indices
S, PH2, Z = 0, 1, 2 # cylindrical coordinate indices
R, H, PH = 0, 1, 2 # spherical coordinate indices
CART, CYL, SPH = 0, 1, 2 # coordinate systems
HYDRO, DM, STAR = 0, 1, 2 # object categories
BIRTH, DEATH = 0, 1 # star log categories
epsilon = 1e-30 # small number

# directories
analysis_dir = "/home/za9132/analysis"
save_dir = analysis_dir + "/figures/current"
sim_base_dir = "/home/za9132/scratch/romain"

# halo cubic fit
halo_poptx = [0.50685686, -0.085267235, 1.9921783, -7.7235762]
halo_popty = [0.5065317, -0.049495054, 1.6965165, -5.8186623]
halo_poptz = [0.49536868, 0.093198848, -1.6449332, 4.735689]

sim_name_to_latex = {
    "alpha_eps0p01": r"$\varepsilon_{\rm SF} = 0.01$", 
    "alpha_eps0p1": r"$\varepsilon_{\rm SF} = 0.1$", 
    "alpha_eps1p0": r"$\varepsilon_{\rm SF} = 1.0$", 
    "eps0p01": r"$\varepsilon_{\rm SF} = 0.01$", 
    "eps0p1": r"$\varepsilon_{\rm SF} = 0.1$", 
    "eps1p0": r"$\varepsilon_{\rm SF} = 1.0$", 
    "alpha_eps0p01_highres": r"$\varepsilon_{\rm SF} = 0.01$ (highres)", 
    "alpha_eps0p1_highres": r"$\varepsilon_{\rm SF} = 0.1$ (highres)", 
    "alpha_eps1p0_highres": r"$\varepsilon_{\rm SF} = 1.0$ (highres)",
    "gas_highres": "Multi-Freefall Model (highres)",
    "dmo": "Dark Matter Only", 
    "gas": "Multi-Freefall Model",
    "alpha_eps0p1_movie": r"$\varepsilon_{\rm SF} = 0.1$ (movie)",
    "mff-zlow": r"Multi-Freefall Model (low $Z$)",
    "mff-zhigh": r"Multi-Freefall Model (high $Z$)",
    "mff-zfunc": r"Multi-Freefall Model (variable $Z$)",
    "mff-dxmin": r"Multi-Freefall Model ($dx_{\rm min}$)",
    "mff-dxloc": r"Multi-Freefall Model ($dx_{\rm loc}$)",
    "bturbfunc": r"Variable $b_{\rm turb}$",
    "bturb0p3": r"$b_{\rm turb} = 0.3$",
    "fiducial": r"fiducial model",
    "bturbfunc_highres": r"Variable $b_{\rm turb}$ (highres)",
    "bturb0p3_highres": r"$b_{\rm turb} = 0.3$ (highres)",
    "bturb1p0_highres": r"$b_{\rm turb} = 1.0$ (highres)",
    "fastfeedback": r"$t_{\rm fbk} = 0.5~{\rm Myr}$",
    "temploc": "Local sound speed",
    "tempfixedcut": "Fixed sound speed with cutoff",
    "bturbfunc_jeans": r"Variable $b_{\rm turb}$ with Jeans refine",
    "mstar0p2": r"$m_* = 0.2$",
    "mstar1p0": r"$m_* = 1.0$",
    "mstar5p0": r"$m_* = 5.0$",
    "mstar0p2cut0p1": r"$m_* = 0.2$, $\varepsilon_{\rm SF} > 0.1$",
    "mstar0p2cut0p01": r"$m_* = 0.2$, $\varepsilon_{\rm SF} > 0.01$",
    "mstar0p2cutphot": r"$m_* = 0.2$, $T < T_{\rm phot}$",
    "eps0p2": r"$\varepsilon_{\rm SF, loc} = 0.2$",
    "eps5p0": r"$\varepsilon_{\rm SF, loc} = 5.0$",
    "instantfeedback": r"$t_{\rm fbk} = 0.0~{\rm Myr}$",
    "epscut": r"$\varepsilon_{\rm SF} > 13$%"
}
