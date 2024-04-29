#!/bin/python3

from matplotlib.pyplot import show
from numpy import deg2rad
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd

e = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test_noisy.ctf"))
q = ebsd.orient.eulers_to_quats(e, "xyz")

ebsd.display.display_quats(q, newaxtitle="original")

u = ebsd.tvflow.denoise(q, weighted=True, beta=0.05)

ebsd.display.display_quats(u, newaxtitle="denoised")

show()
