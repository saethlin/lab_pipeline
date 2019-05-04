import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yt
import caesar
import glob
import h5py

parser = argparse.ArgumentParser('Density Plot')
parser.add_argument('snapshot')
parser.add_argument('caesar')
args = parser.parse_args()
args.caesar = os.path.abspath(args.caesar)

with h5py.File(args.caesar, 'r') as f:
    galaxy_pos = f['galaxy_data/pos'][0]

ds = yt.load(args.snapshot)
center = ds.arr(galaxy_pos, 'kpccm')

image_array = yt.off_axis_projection(
    ds,
    center,
    normal_vector=[0., 1., 0.],
    width=[500., 500., 500.],
    resolution=[4096, 4096],
    item=("gas", "density"),
    north_vector=[0., 0., 1.],
)
image_array = np.rot90(np.log10(image_array))

plt.imshow(
    image_array,
    cmap='magma',
).write_png(f"{os.path.basename(args.snapshot)[:-5]}.png")

