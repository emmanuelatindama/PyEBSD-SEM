#!/bin/python3

from matplotlib.pyplot import show
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd

e = ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/3D_missing_noisy.ctf", missing_phase=0)
q = ebsd.orient.eulers_to_quats(e, "xyz")

ebsd.display.volumetric_displays_quats(
    q, titles=("input, axis 1", "input, axis 2", "input, axis 3")
)

inpainted = ebsd.tvflow.inpaint(q, delta_tolerance=1e-5)
u = ebsd.tvflow.denoise(inpainted, weighted=False, beta=0.05)

ebsd.display.volumetric_displays_quats(
    u, titles=("output, axis 1", "output, axis 2", "output, axis 3")
)

show()
