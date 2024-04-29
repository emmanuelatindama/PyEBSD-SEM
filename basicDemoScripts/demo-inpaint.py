#!/bin/python3

from matplotlib.pyplot import show
from numpy import deg2rad
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd

e = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/missing.ctf", missing_phase=0))
q = ebsd.orient.eulers_to_quats(e, "zxz")

ebsd.display.display_quats(q, newaxtitle="original")

u = ebsd.tvflow.inpaint(q, delta_tolerance=1e-5)

ebsd.display.display_quats(u, newaxtitle="inpainted")

show()
