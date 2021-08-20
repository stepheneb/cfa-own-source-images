#!/usr/local/bin/python3

# https://github.com/esheldon/fitsio

import sys
import os.path
from os import path

import numpy as np
import fitsio
from fitsio import FITS, FITSHDR

if len(sys.argv) < 2:
    print("*** error: no FITS input file directory specified")
    quit()

indir = sys.argv[1]
exists = path.exists(indir)

if exists:
    print("\ninput path exists: " + indir)
else:
    print("\npath not found: " + indir)
    quit()

indir = indir.rstrip("/")

head_tail = os.path.split(indir)

dirname = head_tail[1]
print("dirname: " + dirname)

def fitsinfo(infile):
    print("\nprocessing: "+ infile)

    original_filename = os.path.split(infile)[1]
    print("original_filename: " + original_filename)

    base_filename = os.path.splitext(original_filename)[0]
    print("base_filename: " + base_filename)

    fits=fitsio.FITS(infile)

    print("hdus: " + str(len(fits)))

    img = fits[0].read()

    dtype = str(img.dtype)
    print("img datatype: " + dtype)

    y = len(img)
    x = len(img[0])

    print("x: " + str(x))
    print("y: " + str(y))

    print ("min: " + str(np.min(img)))
    print ("max: " + str(np.max(img)))

    # fig, ax = plt.subplots(2)
    # ax[0].hist(img.flatten(), bins=100, range=(0.1, 40), density=False);
    # ax[0].set_title("M82_Chandra_Xray_mid_energy");
    # ax[1].hist(img.flatten(), bins=100, range=(0.1, 40), density=False);
    # ax[1].set_yscale('log', nonpositive='clip');
    # fig.show()
    # time.sleep(10)



for entry in os.scandir(indir):
    if (entry.path.endswith(".fits") or entry.path.endswith(".FITS")):
        fitsinfo(entry.path)
