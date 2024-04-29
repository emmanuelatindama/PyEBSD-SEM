#!/bin/python3


from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd

from numpy import deg2rad, pi
import matplotlib.pyplot as plt

"""Read noisy ctf file and clean ctf file, then plot the produce ipf-z color plots
"""

clean = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test.ctf"))
noisy = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test_noisy.ctf"))

# ebsd.ipf.plotIPF(clean)
# plt.imshow(clean/2/pi); plt.show()
# ebsd.ipf.plotIPF(noisy)
# plt.imshow(noisy/2/pi); plt.show()


# noisy_5deg = deg2rad(ebsd.orient.add_ebsd_noise(clean, 1))
# ebsd.ipf.plotIPF(noisy_5deg)
# plt.imshow(noisy_5deg/2/pi); plt.show()

clean = deg2rad(ebsd.orient.add_ebsd_noise(clean, std_dev = 2, probability = 0.01))
masked_img, mask, e = ebsd.orient.generate_masked_image(clean, (20,20), masked_reg_val = 1)
plt.imshow(masked_img/2/pi); plt.show()
plt.imshow(mask); plt.show()
plt.imshow(e/2/pi); plt.show()

ebsd.fileio.save_ang_data_as_ctf(filename="/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test_missing20.ctf", angles=e)