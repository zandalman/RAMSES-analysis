'''
Modules used by all files.
'''

import os, subprocess, warnings
from datetime import datetime
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.integrate import quad, quad_vec, trapz, cumtrapz
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.interpolate import griddata, interpn, RegularGridInterpolator
from scipy.signal import welch

from functools import cached_property

import const
