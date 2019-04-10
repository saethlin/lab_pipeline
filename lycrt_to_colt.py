import os
import argparse
import h5py
import numpy as np
import octree
import yt

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("octree", default="octreefile", nargs="?")
parser.add_argument("starfile", default="starfile", nargs="?")
args = parser.parse_args()

ds = yt.load(args.snapshot)

# Convert the lycrt output into an input file for COLT
tree = octree.TREE(args.octree).load()

# load ionization data
# TODO: This is a magic filename that comes from lyrt's internals
# TODO: It would be great if we could have control of that
state_files = sorted([f for f in os.listdir(".") if f.startswith("state")])

state = np.loadtxt(state_files[-1])

x_HI_leaf = state[:, 0]
Jion_leaf = state[:, 1]
scc = tree.sub_cell_check
x_HI = np.zeros(len(scc), dtype=np.float64)
Jion = np.zeros(len(scc), dtype=np.float64)
x_HI[scc == 0] = x_HI_leaf[:]
Jion[scc == 0] = Jion_leaf[:]

stars = np.loadtxt(args.starfile, skiprows=1)
star_positions = stars[:, :3]
L_UV_star = stars[:, 3]

pc = 3.085_677_581_467_192e18  # 1 pc = 3e18 cm
kpc = 1e3 * pc  # 1 kpc = 10^3 pc
Msun = 1.988_435e33  # Solar mass in g

# write to a hdf5 file
with h5py.File(f"converted_{os.path.basename(args.snapshot)}", "w") as f:
    f.attrs["redshift"] = np.float64(ds.current_redshift)
    f.attrs["n_cells"] = np.int32(tree.TOTAL_NUMBER_OF_CELLS)
    f.create_dataset("parent", data=np.array(tree.parent_ID, dtype=np.int32))
    f.create_dataset("child_check", data=np.array(tree.sub_cell_check, dtype=np.int32))
    f.create_dataset("child", data=np.array(tree.sub_cell_IDs, dtype=np.int32))
    f.create_dataset("T", data=np.array(tree.T, dtype=np.float64))  # Temperature (K)
    f["T"].attrs["units"] = b"K"
    # Metallicity (mass fraction)
    f.create_dataset("Z", data=np.array(tree.z, dtype=np.float64))
    f.create_dataset(
        "rho", data=ds.arr(tree.rho, "1e10*Msun/kpc**3").to("g/cm**3"),
        dtype=np.float64,
    )  # Density (g/cm^3)
    f["rho"].attrs["units"] = b"g/cm^3"
    # Neutral fraction n_HI / n_H
    f.create_dataset("x_HI", data=np.array(x_HI, dtype=np.float64))
    # Ionizing intensity in weird units
    f.create_dataset("Jion", data=np.array(Jion, dtype=np.float64))

    f.create_dataset(
        "r", data=ds.arr(tree.min_x, "kpc").to("cm").d, dtype=np.float64
    )  # Minimum corner positions (cm)
    f["r"].attrs["units"] = b"cm"
    f.create_dataset(
        "w", data=ds.arr(tree.width, "kpc").to("cm").d, dtype=np.float64
    )  # Cell widths (cm)
    f["w"].attrs["units"] = b"cm"
    f.create_dataset(
        "v", data=ds.arr(tree.vel, "km/s").to("cm/s"), dtype=np.float64
    )  # Cell velocities (cm/s)
    f["v"].attrs["units"] = b"cm/s"

    f.attrs["n_stars"] = np.int32(star_positions.shape[0])

    # Star positions and luminosities for UV continuum mode
    f.create_dataset(
        "r_star", data=ds.arr(star_positions, "kpc").to("cm").d, dtype=np.float64
    )
    f["r_star"].attrs["units"] = b"cm"

    f.create_dataset("L_UV", data=L_UV_star)
    f["L_UV"].attrs["units"] = b"erg/s/angstrom"
