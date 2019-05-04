import numpy as np
import h5py
import yt
import caesar

c    = 2.99792458e10        # Speed of light in cm/sec
pc   = 3.085677581467192e18 # Units: 1 pc  = 3e18 cm
kpc  = 1.0e3 * pc           # Units: 1 kpc = 3e21 cm
km   = 1.0e5                # Units: 1 km  = 1e5  cm
angstroms = 1.0e-8          # Units: 1 angstrom = 1e-8 cm

snaps = [179]
n_snaps = len(snaps)

verbose = False

# TODO: This is in the same units as star position, whatever that is
R_box = np.array([250.0])

z = np.zeros(n_snaps)
M_star = np.zeros_like(z)
M_star_box = np.zeros_like(z)
#Qion = np.zeros_like(z)
#Qion_box = np.zeros_like(z)
#Lion = np.zeros_like(z)
#Lion_box = np.zeros_like(z)
L_UV = np.zeros_like(z)
L_UV_box = np.zeros_like(z)
L_FUV = np.zeros_like(z)
L_FUV_box = np.zeros_like(z)
L_NUV = np.zeros_like(z)
L_NUV_box = np.zeros_like(z)
M_UV_box = np.zeros_like(z)
beta_box = np.zeros_like(z)

for i in range(n_snaps):
    snap = snaps[i]
    
    snap_dir = "/ufrc/narayanan/kimockb/FIRE2/h113_HR_sn1dy300ro100ss"
    ds = yt.load(f'{snap_dir}/snapshot_{snap:03d}.0.hdf5')
    
    if verbose:
        print('redshift = {}'.format(ds.current_redshift))
        print('hubble   = {}'.format(ds.hubble))
        print('a        = {}'.format(ds.time))
    z[i] = ds.current_redshift

    obj = caesar.load('/ufrc/narayanan/kimockb/FIRE2/h113_HR_sn1dy300ro100ss/Groups/caesar_0179_z1.903.hdf5')
    center = obj.galaxies[0].pos
    halo = obj.halos[0]
    
    data = ds.all_data()
    r_star = data[("PartType4", "Coordinates")]
    star_masses = data[("PartType4", "Masses")]
    star_metallicities = data[("PartType4", "Metallicity_00")]
    star_ages = data[("PartType4", "StellarFormationTime")]
    
    if verbose:
        print('Number of star particles = {}'.format(star_coordinates.size))

    r_star -= center
    radius_star = np.sqrt( np.sum(r_star**2, axis=1) )

    Nvir = 1.
    Rcut = Nvir * halo.radii['virial']
    star_in_halo = (radius_star < Rcut) # star particles within Rcut
    M_star[i] = 1e10 * np.sum(star_masses[star_in_halo])

    star_in_box = (np.abs(r_star[:,0]) < R_box[i]) & (np.abs(r_star[:,1]) < R_box[i]) & (np.abs(r_star[:,2]) < R_box[i]) # star particles within box

    M_star_box[i] = 1e10 * np.sum(star_masses[star_in_box])
    if verbose:
        print('Selection criteria: All particles within {} Rvir = {} kpc'.format(Nvir,Rcut))
        print('Number of star particles in halo = {}'.format(np.count_nonzero(star_in_halo)))
        print('Max star particle distance from halo center = {} kpc'.format(radius_star.max()))
        print('M_star (r<Rcut) = {} Msun'.format(M_star[i]))
        print('M_star (r<Rbox) = {} Msun\n'.format(M_star_box[i]))

    # Finally, calculate ionizing photon production rate for each star particle
    import bpass.bpass as bp # see bpass/bpass.py
#    Qion_star = (1.0e10*p4.m) * 10**bp.compute_stellar_luminosity(p4.age, p4.z,
#                IMF="BPASSv2_imf135_100", BAND_name='ionizing', binary=True)
#    Lion_star = (1.0e10*p4.m) * 10**bp.compute_stellar_luminosity(p4.age, p4.z,
#                IMF="BPASSv2_imf135_100", BAND_name='Lion', binary=True)
    L_UV_star = (1.0e10*star_masses) * 10**bp.compute_stellar_luminosity(star_ages, star_metallicities,
                IMF="BPASSv2_imf135_100", BAND_name='UV', binary=True)
    L_FUV_star = (1.0e10*star_masses) * 10**bp.compute_stellar_luminosity(star_ages, star_metallicities,
                IMF="BPASSv2_imf135_100", BAND_name='FUV', binary=True)
    L_NUV_star = (1.0e10*star_masses) * 10**bp.compute_stellar_luminosity(star_ages, star_metallicities,
                IMF="BPASSv2_imf135_100", BAND_name='NUV', binary=True)
    #Qion[i] = np.sum(Qion_star[star_in_halo])
    #Lion[i] = np.sum(Lion_star[star_in_halo])
    L_UV[i] = np.sum(L_UV_star[star_in_halo])
    L_FUV[i] = np.sum(L_FUV_star[star_in_halo])
    L_NUV[i] = np.sum(L_NUV_star[star_in_halo])
    #Qion_box[i] = np.sum(Qion_star[star_in_box])
    #Lion_box[i] = np.sum(Lion_star[star_in_box])
    L_UV_box[i] = np.sum(L_UV_star[star_in_box])
    L_FUV_box[i] = np.sum(L_FUV_star[star_in_box])
    L_NUV_box[i] = np.sum(L_NUV_star[star_in_box])
    lambda_UV  = bp.get_band_lambda_eff(BAND_name='UV')  # should be close to 1500 A
    lambda_FUV = bp.get_band_lambda_eff(BAND_name='FUV') # should be close to 1540 A
    lambda_NUV = bp.get_band_lambda_eff(BAND_name='NUV') # should be close to 2300 A
    R_10pc = 10. * pc
    flambda_UV_box  = L_UV_box[i]  / (4.*np.pi * R_10pc**2)
    flambda_FUV_box = L_FUV_box[i] / (4.*np.pi * R_10pc**2)
    flambda_NUV_box = L_NUV_box[i] / (4.*np.pi * R_10pc**2)
    fnu_UV_box  = (L_UV_box[i]  / angstroms) * (lambda_UV *angstroms)**2 / (4.*np.pi * c * R_10pc**2)
    fnu_FUV_box = (L_FUV_box[i] / angstroms) * (lambda_FUV*angstroms)**2 / (4.*np.pi * c * R_10pc**2)
    fnu_NUV_box = (L_NUV_box[i] / angstroms) * (lambda_NUV*angstroms)**2 / (4.*np.pi * c * R_10pc**2)
    M_UV_box[i]  = -2.5 * np.log10(fnu_UV_box)  - 48.6
    M_FUV_box = -2.5 * np.log10(fnu_FUV_box) - 48.6
    M_NUV_box = -2.5 * np.log10(fnu_NUV_box) - 48.6
    beta_box[i] = np.log10(flambda_NUV_box/flambda_FUV_box) / np.log(lambda_NUV/lambda_FUV)
    if verbose:
        #print('Qion min/max = [{}, {}] photons/s'.format(Qion_star.min(),Qion_star.max()))
        #print('Total Qion within halo {} = {} photons/s'.format(id0,np.sum(Qion_star[star_in_halo])))
        #print('Lion min/max = [{}, {}] erg/s'.format(Lion_star.min(),Lion_star.max()))
        #print('Total Lion within halo {} = {} erg/s'.format(id0,np.sum(Lion_star[star_in_halo])))
        #print('Qion_star (r<Rcut) = {} Msun'.format(Qion[i]))
        #print('Qion_star (r<Rbox) = {} Msun'.format(Qion_box[i]))
        #print('Lion_star (r<Rcut) = {} Msun'.format(Lion[i]))
        #print('Lion_star (r<Rbox) = {} Msun\n'.format(Lion_box[i]))
        print('L_UV min/max = [{}, {}] erg/s/angstrom'.format(L_UV_star.min(),L_UV_star.max()))
        print('L_UV_star (r<Rcut) = {} erg/s/angstrom'.format(L_UV[i]))
        print('L_UV_star (r<Rbox) = {} erg/s/angstrom'.format(L_UV_box[i]))
        print('L_UV_star * lambda^2 / c = {} erg/s/Hz'.format(L_UV_box[i]*lambda_UV**2/c))
        print('lambda_UV  = {} angstroms'.format(lambda_UV))
        print('lambda_FUV = {} angstroms'.format(lambda_FUV))
        print('lambda_NUV = {} angstroms'.format(lambda_NUV))
        print('M_UV  = {}'.format(M_UV_box[i]))
        print('M_FUV = {}'.format(M_FUV_box))
        print('M_NUV = {}'.format(M_NUV_box))
        print('beta  = {}\n'.format(beta_box[i]))
    # it uses the function "compute_stellar_luminosity" in bpass/bpass.py
    # this function calculate broad-band photometry for stellar populations, 
    # using pre-tabulated tables based on BPASS synthesis model
    #
    # these tables are located at bpass/photometry
    #
    # usage: compute_stellar_luminosity(age_in_Gyr, z_in_frac, IMF="BPASSv2_imf135_100", 
    #                BAND_name='ionizing', binary=True)
    #    age_in_Gyr - stellar age in Gyr
    #    z_in_frac - stellar metallicity in mass fraction
    #    IMF - which IMF? our default is "BPASSv2_imf135_100", a Kroupa IMF 
    #        from 0.1-100Msun, with a high-mass end slope -2.35
    #    BAND_name - which band? for ionizing photons, use 'ionizing'
    #    binary - use single-star or binary model? should use binary by default
    #
    # it returns the ionizing photon production rate for a stellar population, 
    # in unit number_of_photon per second per Msun (in log scale)
    #
    # therefore, the 10** power gives the actually number, also multiply by 1.0e10*p4.m
    # gives the total number of ionizing photons produced by each particle in the simulation
    #
    # the function calls a C shared library at clib/bpass/bpass.so
    # make sure to compile the C code before using (a simple make would do it)
    star_filename = "stars.hdf5"
    # print('File:', data_filename)
    with h5py.File(star_filename,'w') as f:
        f.attrs['n_stars'] = np.int32(len(L_UV_star[star_in_box]))
        f.attrs['redshift'] = np.float64(z[i])
        f.attrs['M_UV'] = np.float64(M_UV_box[i])
        f.attrs['beta'] = np.float64(beta_box[i])
        f.create_dataset('r',     data=r_star[star_in_box,:]/kpc); f['r'].attrs['units']     = b'cm'
        f.create_dataset('L_UV',  data=L_UV_star[star_in_box]);    f['L_UV'].attrs['units']  = b'erg/s/angstrom'
        f.create_dataset('L_FUV', data=L_FUV_star[star_in_box]);   f['L_FUV'].attrs['units'] = b'erg/s/angstrom'
        f.create_dataset('L_NUV', data=L_NUV_star[star_in_box]);   f['L_NUV'].attrs['units'] = b'erg/s/angstrom'


print('z =', z)
print('M_star =', M_star, 'Msun')
print('M_star_box =', M_star_box, 'Msun')
#print('Qion      =', Qion, '1/s')
#print('Qion_box  =', Qion_box, '1/s')
#print('Lion      =', Lion, 'erg/s')
#print('Lion_box  =', Lion_box, 'erg/s')
print('L_UV      =', L_UV,  'erg/s/angstrom')
print('L_UV_box  =', L_UV_box,  'erg/s/angstrom')
print('L_FUV     =', L_FUV, 'erg/s/angstrom')
print('L_FUV_box =', L_FUV_box, 'erg/s/angstrom')
print('L_NUV     =', L_NUV, 'erg/s/angstrom')
print('L_NUV_box =', L_NUV_box, 'erg/s/angstrom')
print('M_UV      =', M_UV_box)
print('beta      =', beta_box)

