''' Modules used by all files. '''

import os, subprocess, warnings
from datetime import datetime
from tabulate import tabulate
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from astropy.io import ascii

from scipy.integrate import quad, quad_vec, trapz, cumtrapz
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.interpolate import griddata, interpn, RegularGridInterpolator, interp1d
from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.io import FortranFile
from scipy.signal import welch

from functools import cached_property

import const
from config import *
from functions import *
