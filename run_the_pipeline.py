import os
import argparse

import h5py
import numpy as np
import yt
import octree
import bpass
import lycrt
import numexpr


parser = argparse.ArgumentParser(description="THE CLASP")
parser.add_argument('snapshot')
parser.add_argument('--caesar', help="Path to the snapshot's CAESAR file")
parser.add_argument('--outdir', help="Directory to place output files")

args = parser.parse_args()

snap = args.snapshot
caesar_path = args.caesar or os.path.join(os.path.dirname(snap), f'caesar_{os.path.basename(snap)}')
output_dir = args.outdir or '.'


caesar_files = sorted([f.path for f in os.scandir(os.path.dirname(caesar_path)) if 'caesar_00' in f.name])

galaxy_pos = h5py.File(caesar_files[0])['galaxy_data']['pos'][1]

for f in caesar_files:
    pos = h5py.File(f)['galaxy_data']['pos'][:20]
    distance = np.sum((pos - galaxy_pos)**2, axis=1)
    galaxy_pos = pos[np.argmin(distance)]
    print(galaxy_pos)

    if f == caesar_path:
        break


ds = yt.load(snap)
SIZE = ds.quan(250.0, 'kpccm')

galaxy_pos = yt.YTArray(galaxy_pos, 'kpccm', ds.unit_registry)

f = h5py.File(args.snapshot)

gas_position = yt.YTArray(f['PartType0/Coordinates'][:], 'code_length', ds.unit_registry).convert_to_units('kpccm')

lhs = numexpr.evaluate('sum((gas_position - galaxy_pos)**2, axis=1)')
glist = numexpr.evaluate('lhs < (2 * SIZE)**2')
assert np.sum(glist) > 0

# Filter the gas particles
gas_position = gas_position[glist]
velocity = f['PartType0/Velocities'][:][glist]
mass = f['PartType0/Masses'][glist]
internal_energy = f['PartType0/InternalEnergy'][glist]
neutral_fraction = f['PartType0/NeutralHydrogenAbundance'][glist]
electron_fraction = f['PartType0/ElectronAbundance'][glist]
smoothing_length = f['PartType0/SmoothingLength'][glist]
gas_metallicity = f['PartType0/Metallicity'][:, 0][glist]

# Filter the star particles
star_position = yt.YTArray(f['PartType4/Coordinates'][:], 'code_length', ds.unit_registry).convert_to_units('kpccm')
lhs = numexpr.evaluate('sum((star_position - galaxy_pos)**2, axis=1)')
slist = numexpr.evaluate('lhs < (2 * SIZE)**2')

star_position = star_position[slist]
star_metallicity = f['PartType4/Metallicity'][:, 0][slist]

scalefactor = f['PartType4/StellarFormationTime'][slist]
formation_z = (1. / scalefactor) - 1.
yt_cosmo = yt.utilities.cosmology.Cosmology()
stellar_formation_age = yt_cosmo.t_from_z(formation_z).in_units('Gyr')
# Age of the universe right now
simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units('Gyr')
stellar_ages = (simtime - stellar_formation_age).in_units('Gyr')

log_luminosity = bpass.compute_stellar_luminosity(stellar_ages, star_metallicity)

"""
Send the star and gas data to lycrt via files
"""
with open(os.path.join(args.outdir, 'starfile'), 'w') as starfile, open(os.path.join(args.outdir, 'paramfile'), 'w') as paramfile, open(os.path.join(args.outdir, 'octreefile'), 'w') as octreefile:

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
        fname=octreefile.name.encode('utf-8'),
        MAX_DISTANCE_FROM0=SIZE,
        TREE_MAX_NPART=1,
        TREE_MIN_SIZE=0.01)

    starfile.write('{}\n'.format(star_metallicity.size))
    for (pos, lum) in zip(star_position, log_luminosity):
        pos -= galaxy_pos
        starfile.write('{} {} {} {}\n'.format(pos[0], pos[1], pos[2], lum))
    starfile.flush()

    paramfile.write("""octree_file     {}
star_file       {}
n_iters         10
n_photons_star  1e5
n_photons_uvb   1e5
J_uvb           1.2082e-22
dust_kappa      1
n_mu            1
n_phi           1""".format(octreefile.name, starfile.name))
    paramfile.flush()

    lycrt.lycrt(paramfile.name)

    # Convert the lycrt output into an input file for COLT
    tree = octree.TREE(octreefile.name).load()

    # load ionization data
    # TODO: This is a magic filename that comes from lyrt's internals
    # TODO: It would be great if we could have control of that
    state = np.loadtxt('state_09.txt')
    x_HI_leaf = state[:, 0]
    Jion_leaf = state[:, 1]
    scc = tree.sub_cell_check
    x_HI = np.zeros(len(scc), dtype=np.float64)
    Jion = np.zeros(len(scc), dtype=np.float64)
    x_HI[scc == 0] = x_HI_leaf[:]
    Jion[scc == 0] = Jion_leaf[:]

    km = 1.0e5                  # 1 km = 10^5 cm
    pc = 3.085677581467192e18   # 1 pc = 3e18 cm
    kpc = 1e3 * pc              # 1 kpc = 10^3 pc
    Mpc = 1e6 * pc              # 1 Mpc = 10^6 pc
    Msun = 1.988435e33          # Solar mass in g

    # write to a hdf5 file
    f = h5py.File(os.path.join(output_dir, f'converted_{os.path.basename(snap)}'), 'w')
    f.attrs['redshift'] = np.float64(ds.current_redshift)
    f.attrs['n_cells'] = np.int32(tree.TOTAL_NUMBER_OF_CELLS)
    f.create_dataset('parent', data=np.array(tree.parent_ID, dtype=np.int32))
    f.create_dataset(
        'child_check',
        data=np.array(
            tree.sub_cell_check,
            dtype=np.int32))
    f.create_dataset('child', data=np.array(tree.sub_cell_IDs, dtype=np.int32))
    f.create_dataset(
        'T',
        data=np.array(
            tree.T,
            dtype=np.float64))  # Temperature (K)
    f['T'].attrs['units'] = b'K'
    # Metallicity (mass fraction)
    f.create_dataset('Z', data=np.array(tree.z, dtype=np.float64))
    f.create_dataset(
        'rho',
        data=np.array(
            1e10 *
            Msun /
            kpc**3 *
            tree.rho,
            dtype=np.float64))  # Density (g/cm^3)
    f['rho'].attrs['units'] = b'g/cm^3'
    # Neutral fraction n_HI / n_H
    f.create_dataset('x_HI', data=np.array(x_HI, dtype=np.float64))
    # Ionizing intensity in weird units
    f.create_dataset('Jion', data=np.array(Jion, dtype=np.float64))
    f.create_dataset(
        'r',
        data=np.array(
            kpc *
            tree.min_x,
            dtype=np.float64))  # Minimum corner positions (cm)
    f['r'].attrs['units'] = b'cm'
    f.create_dataset(
        'w',
        data=np.array(
            kpc * tree.width,
            dtype=np.float64))  # Cell widths (cm)
    f['w'].attrs['units'] = b'cm'
    f.create_dataset(
        'v',
        data=np.array(
            km * tree.vel,
            dtype=np.float64))  # Cell velocities (cm/s)
    f['v'].attrs['units'] = b'cm/s'
    f.close()

