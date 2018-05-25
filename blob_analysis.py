import os
import tempfile

import caesar
import yt
import octree
import bpass
import lycrt

# snap = 'snapshot_135.hdf5'
snap = 'lycrt-example/L43_ref10_HL00230_ref12_2/snapshot_067.hdf5'
ds = yt.load(snap)
data = ds.all_data()


caesarname = os.path.join(os.path.dirname(snap), 'caesar_'+os.path.basename(snap))
try:
    obj = caesar.load(caesarname)
except IOError:
    obj = caesar.CAESAR(ds)
    obj.member_search()#unbind_halos=True, unbind_galaxies=False)
    obj.save(caesarname)

# Grab the large but uncontaminated halo
halo = obj.halos[0]

# Select based on the virial radius of the selected halo
virial_radius = halo.radii['virial']
gas_position = data[('PartType0', 'Coordinates')]
glist = ((gas_position - halo.pos)**2).sum(axis=1) < virial_radius**2

#print("Position: {}".format(halo.pos))
#print("Virial Radius: {}".format(halo.radii['virial']))
#exit()

gas_position = data[('PartType0', 'Coordinates')][glist]
mass = data[('PartType0', 'Masses')][glist]
velocity = data[('PartType0', 'Velocities')][glist]
internal_energy = data[('PartType0', 'InternalEnergy')][glist]
neutral_fraction = data[('PartType0', 'NeutralHydrogenAbundance')][glist]
electron_fraction = data[('PartType0', 'ElectronAbundance')][glist]
gas_metallicity = data[('PartType0', 'Metallicity_00')][glist]
smoothing_length = data[('PartType0', 'SmoothingLength')][glist]


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

# Send the star and gas data to lycrt via temp files
with tempfile.NamedTemporaryFile() as starfile, tempfile.NamedTemporaryFile() as octreefile, tempfile.NamedTemporaryFile() as paramfile:

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
        fname=octreefile.name)

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

