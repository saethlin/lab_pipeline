import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import h5py
import os

for colt_path in sorted(glob.glob("lycrt_runs/**/tot_SMC.h5")):
    png_path = os.path.join(os.path.dirname(colt_path), 'picture.png')

    image = h5py.File(colt_path)['LOS/SB'][0]

    max_clip = 1e-14
    min_clip = 1e-23
    image = image.clip(min_clip, max_clip)

    plt.imshow(
        image,
        cmap="magma",
        norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
        extent=[-125, 125, -125, 125],
    ).write_png(png_path)

    plt.clf()

