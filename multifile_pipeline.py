import os
import argparse

import h5py
import numpy as np
import yt
import octree
import bpass
import lycrt
import numexpr


def load_from_files(files, dataset_path, masks):
    chunks = []
    for (f, m) in zip(files, masks):
        if f[dataset_path].ndim == 2:
            chunk = f[dataset_path][:][m]
        else:
            chunk = f[dataset_path][m]

        if chunk.size > 0:
            chunks.append(chunk)

    return np.concatenate(chunks, axis=0)


parser = argparse.ArgumentParser(description="THE CLASP")
parser.add_argument("snapshots", nargs="+")
parser.add_argument("--caesar", help="Path to the snapshot's CAESAR file")
parser.add_argument("--outdir", help="Directory to place output files")

args = parser.parse_args()
args.outdir = args.outdir or "."

ds = yt.load(args.snapshots[0])
SIZE = ds.quan(250.0, "kpccm")

galaxy_pos = h5py.File(args.caesar)["galaxy_data"]["pos"][0]
galaxy_pos = yt.YTArray(galaxy_pos, "kpccm", ds.unit_registry)

# This is disgusting but there's no standard file extension, and people put other crap in their snapdirs
snap_files = []
for f in args.snapshots:
    try:
        snap_files.append(h5py.File(f))
    except OSError:
        pass

masks = []
chunks = []
for f in snap_files:
    gas_position = f["PartType0/Coordinates"][:]
    gas_position = yt.YTArray(
        gas_position, "code_length", ds.unit_registry
    ).convert_to_units("kpccm")
    glist = np.all(np.abs(gas_position - galaxy_pos) < SIZE, axis=1)
    masks.append(glist)
    chunks.append(gas_position[glist])

gas_position = yt.YTArray(np.concatenate(chunks, axis=0), "kpccm", ds.unit_registry)

# Filter the gas particles
velocity = load_from_files(snap_files, "PartType0/Velocities", masks)
mass = load_from_files(snap_files, "PartType0/Masses", masks)
internal_energy = load_from_files(snap_files, "PartType0/InternalEnergy", masks)
neutral_fraction = load_from_files(
    snap_files, "PartType0/NeutralHydrogenAbundance", masks
)
electron_fraction = load_from_files(snap_files, "PartType0/ElectronAbundance", masks)
smoothing_length = load_from_files(snap_files, "PartType0/SmoothingLength", masks)

# Metallicity is a special case, because we only want the first kind
gas_metallicity = np.concatenate(
    [f["PartType0/Metallicity"][:, 0][m] for (f, m) in zip(snap_files, masks)], axis=0
)


# Filter the star particles
masks = []
chunks = []
for f in snap_files:
    star_position = f["PartType4/Coordinates"][:]
    star_position = yt.YTArray(
        star_position, "code_length", ds.unit_registry
    ).convert_to_units("kpccm")
    slist = np.all(np.abs(star_position - galaxy_pos) < SIZE, axis=1)
    masks.append(slist)
    chunks.append(star_position[slist])

star_position = yt.YTArray(np.concatenate(chunks, axis=0), "kpccm", ds.unit_registry)

star_metallicity = np.concatenate(
    [f["PartType4/Metallicity"][:, 0][m] for (f, m) in zip(snap_files, masks)], axis=0
)
scalefactor = load_from_files(snap_files, "PartType4/StellarFormationTime", masks)
star_masses = load_from_files(snap_files, "PartType4/Masses", masks)

formation_z = (1. / scalefactor) - 1.
yt_cosmo = yt.utilities.cosmology.Cosmology()
stellar_formation_age = yt_cosmo.t_from_z(formation_z).in_units("Gyr")
# Age of the universe right now
simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
stellar_ages = (simtime - stellar_formation_age).in_units("Gyr")

log_luminosity = bpass.compute_stellar_luminosity(
    stellar_ages, star_metallicity, binary=True
)


# Entries for the stars file
# TODO: Is this a duplicate of the above call?
# TODO: What are the required input units here?
L_UV_star = (1.0e10 * star_masses) * 10 ** bpass.compute_stellar_luminosity(
    stellar_ages, star_metallicity, BAND_name="UV", binary=True
)


"""
Send the star and gas data to lycrt via files
"""
with open(os.path.join(args.outdir, "starfile"), "w") as starfile, open(
    os.path.join(args.outdir, "paramfile"), "w"
) as paramfile, open(os.path.join(args.outdir, "octreefile"), "w") as octreefile:

    octree.build_octree_from_particle(
        pos=gas_position,
        vel=velocity,
        m=mass,
        h=smoothing_length,
        nh=neutral_fraction,
        ne=electron_fraction,
        u=internal_energy,
        z=gas_metallicity,
        cen=galaxy_pos,
        fname=octreefile.name.encode("utf-8"),
        MAX_DISTANCE_FROM0=SIZE,
        TREE_MAX_NPART=1,
        TREE_MIN_SIZE=0.01,
    )

    starfile.write("{}\n".format(star_metallicity.size))
    for (pos, lum) in zip(star_position, log_luminosity):
        pos -= galaxy_pos
        starfile.write("{} {} {} {}\n".format(pos[0], pos[1], pos[2], lum))
    starfile.flush()

    paramfile.write(
        """octree_file     {}
star_file       {}
n_iters         10
n_photons_star  1e6
n_photons_uvb   1e6
J_uvb           1.2082e-22
dust_kappa      1
n_mu            1
n_phi           1""".format(
            octreefile.name, starfile.name
        )
    )
    paramfile.flush()
    # original J_uvb value was 1.2082e-22
    lycrt.lycrt(paramfile.name)

    # Convert the lycrt output into an input file for COLT
    tree = octree.TREE(octreefile.name).load()

    # load ionization data
    # TODO: This is a magic filename that comes from lyrt's internals
    # TODO: It would be great if we could have control of that
    # Grab the last 5 state reports and average them
    state_files = sorted([f for f in os.listdir(".") if f.startswith("state")])[-5:]
    state = np.loadtxt(state_files[0])
    for f in state_files[1:]:
        state += np.loadtxt(f)
    state = state / 5.0

    x_HI_leaf = state[:, 0]
    Jion_leaf = state[:, 1]
    scc = tree.sub_cell_check
    x_HI = np.zeros(len(scc), dtype=np.float64)
    Jion = np.zeros(len(scc), dtype=np.float64)
    x_HI[scc == 0] = x_HI_leaf[:]
    Jion[scc == 0] = Jion_leaf[:]

    km = 1.0e5  # 1 km = 10^5 cm
    pc = 3.085677581467192e18  # 1 pc = 3e18 cm
    kpc = 1e3 * pc  # 1 kpc = 10^3 pc
    Mpc = 1e6 * pc  # 1 Mpc = 10^6 pc
    Msun = 1.988435e33  # Solar mass in g

    # write to a hdf5 file
    f = h5py.File(
        os.path.join(args.outdir, f"converted_{os.path.basename(args.snapshots[0])}"),
        "w",
    )
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
        "rho", data=np.array(1e10 * Msun / kpc ** 3 * tree.rho, dtype=np.float64)
    )  # Density (g/cm^3)
    f["rho"].attrs["units"] = b"g/cm^3"
    # Neutral fraction n_HI / n_H
    f.create_dataset("x_HI", data=np.array(x_HI, dtype=np.float64))
    # Ionizing intensity in weird units
    f.create_dataset("Jion", data=np.array(Jion, dtype=np.float64))
    f.create_dataset(
        "r", data=np.array(kpc * tree.min_x, dtype=np.float64)
    )  # Minimum corner positions (cm)
    f["r"].attrs["units"] = b"cm"
    f.create_dataset(
        "w", data=np.array(kpc * tree.width, dtype=np.float64)
    )  # Cell widths (cm)
    f["w"].attrs["units"] = b"cm"
    f.create_dataset(
        "v", data=np.array(km * tree.vel, dtype=np.float64)
    )  # Cell velocities (cm/s)
    f["v"].attrs["units"] = b"cm/s"

    # Star positions, for UV continuum calculation
    f.attrs["n_stars"] = np.int32(star_positions.size)

    f.create_dataset("r_star", data=star_positions / kpc)
    f["r_star"].attrs["units"] = b"cm/s"

    f.create_dataset("L_UV", data=L_UV_star)
    f["L_UV"].attrs["units"] = b"erg/s/angstrom"

    f.close()
