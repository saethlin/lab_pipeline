import os
import argparse
import yt
import caesar
import octree
import bpass
import numpy as np
from scipy.interpolate import interp1d
import powderday.agn_models
import powderday.config

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("caesar", help="Path to the snapshot's CAESAR file")
parser.add_argument("--outdir",
                    help="Directory to place output files",
                    default=".")
parser.add_argument('--agn_model',
                    choices=['none', 'hopkins', 'nenkova'],
                    default='none')
parser.add_argument("--n_iters",
                    help="Number of lycrt iterations",
                    type=int,
                    default=50)
parser.add_argument("--galaxy_index",
                    help="CAESAR galaxy index",
                    type=int,
                    default=0)
parser.add_argument("--cube_size",
                    help="Side length of output cube (kpc)",
                    type=float,
                    default=75.)
parser.add_argument("--max_npart",
                    help="Max number of particles in an octree cell",
                    type=int,
                    default=1)
parser.add_argument(
    "--min_size",
    help="Minimum size of an octree cell (kpc)",
    type=float,
    default=0.01,
)
parser.add_argument(
    '--temperature_multiplier',
    type=float,
    default=1.0,
)
parser.add_argument(
    '--agn_multiplier',
    type=float,
    default=1.0,
)
parser.add_argument(
    '--agn_luminosity',
    choices=['eddington', 'magorrian'],
    default='eddington',
)
parser.add_argument(
    "--n_photons_star",
    help="Number of star photons to simulate",
    type=float,
    default=1e7,
)
args = parser.parse_args()
print(args)

os.makedirs(args.outdir, exist_ok=True)

ds = yt.load(args.snapshot)
args.min_size = ds.quan(args.min_size, 'kpc')
SIZE = ds.quan(args.cube_size, "kpc")

fg_table = np.loadtxt(
    '/ufrc/narayanan/kimockb/lycrt-example/model/fg_uvb_table.txt')
j_uvb = interp1d(fg_table[:, 0], fg_table[:, 1] * 1e-21)(ds.current_redshift)

# Assume the galaxy of interest is the largest
obj = caesar.quick_load(args.caesar)
galaxy_pos = obj.galaxies[args.galaxy_index].pos
region = ds.box(galaxy_pos - SIZE, galaxy_pos + SIZE)


const = yt.physical_constants
if args.agn_luminosity == 'magorrian' and args.agn_model != 'none':
    agn_pos = []
    agn_luminosity = []
    for gal in obj.galaxies:
        if np.all(np.abs((gal.pos - galaxy_pos).to('kpc')) < SIZE):

            agn_pos.append(gal.pos.copy().to('kpc'))
            total_luminosity = 4 * np.pi * const.gravitational_constant * const.amu * const.c * gal.masses[
                'stellar'] * 0.0006 / const.thompson_cross_section

            if args.agn_model == 'hopkins':
                nu, lum = powderday.agn_models.hopkins.agn_spectrum(
                    np.log10(total_luminosity.to('Lsun')))
                nu = ds.arr(10**nu, 'Hz')
                lum = ds.arr(10**lum, 'erg/s')

            elif args.agn_model == 'nenkova':
                powderday.config.par = type(powderday)('parameters')
                powderday.config.par.BH_modelfile = "/ufrc/narayanan/kimockb/clumpy_models_201410_tvavg.hdf5"
                nu, lum = powderday.agn_models.nenkova.Nenkova2008(
                ).agn_spectrum(np.log10(total_luminosity.to('Lsun')))
                nu = ds.arr(10**nu, 'Hz')
                lum = ds.arr(10**lum, 'erg/s')

            nu = nu[:-4]
            lum = lum[:-4]
            idx = np.argsort(nu)
            nu = nu[idx]
            lum = lum[idx]

            lum_photons = lum / (nu * const.planck_constant)
            mask = (nu >
                    (yt.YTQuantity(13.6, 'eV') / const.planck_constant)) & (
                        nu < yt.YTQuantity(24.4, 'eV') / const.planck_constant)
            agn_luminosity.append(np.trapz(lum_photons[mask], nu[mask]).d)

    agn_luminosity = np.array(agn_luminosity)
    agn_luminosity *= args.agn_multiplier
elif args.agn_luminosity == 'eddington' and args.agn_model != 'none':
    agn_pos = region[('PartType5', 'Coordinates')].to('kpc')
    agn_mass = region[('PartType5', 'BH_Mass')].to('Msun')

    agn_luminosity = []

    for i in range(len(agn_mass)):
        if agn_mass[i] == 0.0:
            agn_luminosity.append(0.0)
            continue

        total_luminosity = 4 * np.pi * const.gravitational_constant * const.amu * const.c * agn_mass[
            i] / const.thompson_cross_section

        if args.agn_model == 'hopkins':
            nu, lum = powderday.agn_models.hopkins.agn_spectrum(
                np.log10(total_luminosity.to('Lsun')))
            nu = ds.arr(10**nu, 'Hz')
            lum = ds.arr(10**lum, 'erg/s')

        elif args.agn_model == 'nenkova':
            powderday.config.par = type(powderday)('parameters')
            powderday.config.par.BH_modelfile = "/ufrc/narayanan/kimockb/clumpy_models_201410_tvavg.hdf5"
            nu, lum = powderday.agn_models.nenkova.Nenkova2008(
            ).agn_spectrum(np.log10(total_luminosity.to('Lsun')))
            nu = ds.arr(10**nu, 'Hz')
            lum = ds.arr(10**lum, 'erg/s')

        nu = nu[:-4]
        lum = lum[:-4]
        idx = np.argsort(nu)
        nu = nu[idx]
        lum = lum[idx]

        lum_photons = lum / (nu * const.planck_constant)
        mask = (nu >
                (yt.YTQuantity(13.6, 'eV') / const.planck_constant)) & (
                    nu < yt.YTQuantity(24.4, 'eV') / const.planck_constant)
        agn_luminosity.append(np.trapz(lum_photons[mask], nu[mask]).d)

    agn_luminosity = np.array(agn_luminosity)
    agn_luminosity *= args.agn_multiplier

# Data we need from the gas particles
gas_position = region[("PartType0", "Coordinates")]
velocity = region[("PartType0", "Velocities")] - obj.halos[0].vel
mass = region[("PartType0", "Masses")]
internal_energy = region[("PartType0",
                          "InternalEnergy")] * args.temperature_multiplier
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
star_ionizing_luminosity = star_masses.to(
    'Msun') * 10**bpass.compute_stellar_luminosity(
        stellar_ages, star_metallicity, BAND_name="ionizing")

L_UV_star = star_masses.to('Msun') * 10**bpass.compute_stellar_luminosity(
    stellar_ages, star_metallicity, BAND_name="UV")
"""
Send the star and gas data to lycrt via files
"""
# ascale = ds.scale_factor
# hinv = 1/ds.hubble_constant
with open(os.path.join(args.outdir, "starfile"), "w") as starfile, open(
        os.path.join(args.outdir, "paramfile"),
        "w") as paramfile, open(os.path.join(args.outdir, "octreefile"),
                                "w") as octreefile:

    if args.agn_model != 'none':
        starfile.write("{}\n".format(star_positions.shape[0] +
                                     len(agn_luminosity)))
        for pos, lum in zip(agn_pos, agn_luminosity):
            centered_pos = (pos - galaxy_pos).to('kpc').d
            starfile.write("{} {} {} {}\n".format(*centered_pos, lum))
    else:
        starfile.write("{}\n".format(star_positions.shape[0]))

    for (pos, lum) in zip((star_positions - galaxy_pos).to('kpc').d,
                          star_ionizing_luminosity.d):
        starfile.write("{} {} {} {}\n".format(pos[0], pos[1], pos[2], lum))
    starfile.flush()

    paramfile.write("""octree_file     {}
star_file       {}
n_iters         {}
n_photons_star  1e7
n_photons_uvb   1e7
J_uvb           {}
dust_kappa      1
n_mu            1
n_phi           1""".format(
        octreefile.name,
        starfile.name,
        args.n_iters,
        j_uvb,
    ))
    paramfile.flush()

    octree.build_octree_from_particle(
        pos=gas_position.to('kpc').d,
        vel=velocity.to('km/s').d,
        m=mass.to('1e10*Msun').d,
        h=smoothing_length.to('kpc').d,
        nh=neutral_fraction.d,
        ne=electron_fraction.d,
        u=internal_energy.d,
        z=gas_metallicity.d,
        cen=galaxy_pos.to('kpc').d,
        fname=octreefile.name.encode("utf-8"),
        MAX_DISTANCE_FROM0=SIZE.to('kpc').d,
        TREE_MAX_NPART=args.max_npart,
        TREE_MIN_SIZE=args.min_size.to('kpc').d,
    )
