import os
import tempfile
import argparse

import h5py
import numpy as np
import caesar
import yt
import octree
import bpass
import lycrt

parser = argparse.ArgumentParser(description="THE CLASP")
parser.add_argument('snapshot', type=str)

snap = parser.parse_args().snapshot

ds = yt.load(snap)
data = ds.all_data()

"""
Use CAESAR to extract the high-resolution halo 
"""

# Cache the caesar file next to the original- should probably put them local
caesarname = os.path.join(os.path.dirname(snap), 'caesar_'+os.path.basename(snap))
try:
    obj = caesar.load(caesarname)
except IOError:
    obj = caesar.CAESAR(ds)
    obj.member_search()#unbind_halos=True, unbind_galaxies=False)
    #obj.save(caesarname)

# Grab the large but uncontaminated halo
halo = obj.halos[0]

# Select based on the virial radius of the selected halo
virial_radius = halo.radii['virial']
gas_position = data[('PartType0', 'Coordinates')]
glist = ((gas_position - halo.pos)**2).sum(axis=1) < virial_radius**2

gas_position = data[('PartType0', 'Coordinates')][glist]
mass = data[('PartType0', 'Masses')][glist]
velocity = data[('PartType0', 'Velocities')][glist]
internal_energy = data[('PartType0', 'InternalEnergy')][glist]
neutral_fraction = data[('PartType0', 'NeutralHydrogenAbundance')][glist]
electron_fraction = data[('PartType0', 'ElectronAbundance')][glist]
gas_metallicity = data[('PartType0', 'Metallicity_00')][glist]
smoothing_length = data[('PartType0', 'SmoothingLength')][glist]

"""
Prepare inputs for xiangcheng's BPASS wrapper
"""
# Grab the required star data and run it through bpass
star_position = data[('PartType4', 'Coordinates')]
slist = ((star_position - halo.pos)**2).sum(axis=1) < virial_radius**2

star_metallicity = data[('PartType4', 'Metallicity_00')][slist]
star_position = data[('PartType4', 'Coordinates')][slist]

# Formation time is given in redshift at formation, convert to Gyr
scalefactor = data[('PartType4', 'StellarFormationTime')][slist]
formation_z = (1. / scalefactor) - 1.
yt_cosmo = yt.utilities.cosmology.Cosmology()
stellar_formation_age = yt_cosmo.t_from_z(formation_z).in_units('Gyr')
# Age of the universe right now
simtime = yt_cosmo.t_from_z(obj.simulation.redshift).in_units('Gyr')
stellar_ages = (simtime - stellar_formation_age).in_units('Gyr')

log_luminosity = bpass.compute_stellar_luminosity(stellar_ages, star_metallicity)

"""
Send the star and gas data to lycrt via temp files
"""
with tempfile.NamedTemporaryFile(mode='w') as starfile, tempfile.NamedTemporaryFile(mode='w') as paramfile, tempfile.NamedTemporaryFile(mode='w') as octreefile:

    octree.build_octree_from_particle(
        pos=gas_position,
        vel=velocity,
        m=mass,
        h=smoothing_length,
        nh=neutral_fraction,
        ne=electron_fraction,
        u=internal_energy,
        z=gas_metallicity,
        cen=halo.pos,
        fname=octreefile.name.encode('utf-8'))

    starfile.write('{}\n'.format(star_metallicity.size))
    for (pos, lum) in zip(star_position, log_luminosity):
        starfile.write('{} {} {} {}'.format(pos[0], pos[1], pos[2], lum))
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
    f = h5py.File('converted_' + os.path.basename(snap), 'w')
    f.attrs['redshift'] = 5.0
    #f.attrs['redshift'] = np.float64(sp.redshift)
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

# Now we can finally kick off COLT

