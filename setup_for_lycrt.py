import os
import argparse
import yt
import caesar
import octree
import bpass
import numpy as np

# Tell Anaconda MKL to not launch a bajillion threads and slow us down
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("--caesar", help="Path to the snapshot's CAESAR file")
parser.add_argument("--outdir", help="Directory to place output files", default=".")
parser.add_argument(
    "--n_iters", help="Number of lycrt iterations", type=int, default=20
)
parser.add_argument("--galaxy_index", help="CAESAR galaxy index", type=int, default=0)
parser.add_argument(
    "--cube_size", help="Side length of output cube (kpccm)", type=float, default=250.
)
parser.add_argument(
    "--max_npart", help="Max number of particles in an octree cell", type=int, default=1
)
parser.add_argument(
    "--min_size",
    help="Minimum size of an octree cell (kpc)",
    type=float,
    default=0.01,
)

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

ds = yt.load(args.snapshot)
args.min_size = ds.quan(args.min_size, 'kpc')
SIZE = ds.quan(args.cube_size, "kpccm")

from scipy.interpolate import interp1d
fg_table = np.loadtxt('/ufrc/narayanan/kimockb/lycrt-example/model/fg_uvb_table.txt')
j_uvb = interp1d(fg_table[:, 0], fg_table[:, 1]*1e-21)(ds.current_redshift)

# Assume the galaxy of interest is the largest
obj = caesar.load(args.caesar, load_limit=10)
galaxy_pos = obj.galaxies[args.galaxy_index].pos
region = ds.box(galaxy_pos - SIZE, galaxy_pos + SIZE)

# Data we need from the gas particles
gas_position = region[("PartType0", "Coordinates")]
velocity = region[("PartType0", "Velocities")] - obj.galaxies[0].vel
mass = region[("PartType0", "Masses")]
internal_energy = region[("PartType0", "InternalEnergy")]
neutral_fraction = region[("PartType0", "NeutralHydrogenAbundance")]
electron_fraction = region[("PartType0", "ElectronAbundance")]
smoothing_length = region[("PartType0", "SmoothingLength")]
gas_metallicity = region[("PartType0", "Metallicity_00")]

# Star particles
star_positions = region[("PartType4", "Coordinates")]
star_metallicity = region[("PartType4", "Metallicity_00")]
scalefactor = region[("PartType4", "StellarFormationTime")]
star_masses = region[("PartType4", "Masses")]

# Compute the age of all the star particles from the provided scale factor at creation
formation_z = (1.0 / scalefactor) - 1.0
yt_cosmo = yt.utilities.cosmology.Cosmology()
stellar_formation_age = yt_cosmo.t_from_z(formation_z).in_units("Gyr")
# Age of the universe right now
simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
stellar_ages = (simtime - stellar_formation_age).in_units("Gyr")

# TODO: Still unclear if my usage of this function is correct
star_ionizing_luminosity = star_masses.to('Msun') * 10 ** bpass.compute_stellar_luminosity(
    stellar_ages, star_metallicity, BAND_name="ionizing"
)

L_UV_star = star_masses.to('Msun') * 10 ** bpass.compute_stellar_luminosity(
    stellar_ages, star_metallicity, BAND_name="UV"
)


"""
Send the star and gas data to lycrt via files
"""
#ascale = ds.scale_factor
#hinv = 1/ds.hubble_constant
with open(os.path.join(args.outdir, "starfile"), "w") as starfile, open(
    os.path.join(args.outdir, "paramfile"), "w"
) as paramfile, open(os.path.join(args.outdir, "octreefile"), "w") as octreefile:

    octree.build_octree_from_particle(
        #pos=gas_position * ascale * hinv,
        pos=gas_position.to('kpc').d,
        #vel=velocity * np.sqrt(ascale),
        vel=velocity.to('km/s').d,
        #m=mass * hinv,
        m=mass.to('Msun').d/1e10,
        h=smoothing_length.d,
        nh=neutral_fraction.d,
        ne=electron_fraction.d,
        u=internal_energy.d,
        z=gas_metallicity.d,
        #cen=galaxy_pos.to('code_length') * ascale * hinv,
        cen=galaxy_pos.to('kpc').d,
        fname=octreefile.name.encode("utf-8"),
        #MAX_DISTANCE_FROM0=SIZE.to('code_length') * ascale * hinv,
        MAX_DISTANCE_FROM0=SIZE.to('kpc').d,
        TREE_MAX_NPART=args.max_npart,
        TREE_MIN_SIZE=args.min_size.to('kpc').d,
    )

    starfile.write("{}\n".format(star_metallicity.size))
    for (pos, lum) in zip((star_positions - galaxy_pos).convert_to_units('kpc').d, star_ionizing_luminosity.d):
        starfile.write("{} {} {} {}\n".format(pos[0], pos[1], pos[2], lum))

    paramfile.write(
        """octree_file     {}
star_file       {}
n_iters         {}
n_photons_star  1e7
n_photons_uvb   1e7
J_uvb           {}
dust_kappa      1
n_mu            1
n_phi           1""".format(
            octreefile.name, starfile.name, args.n_iters, j_uvb,
        )
    )
