import h5py
import numpy as np
import glob
import shutil

km = 1.0e5
angstroms = 1.0e-8
arcsec = 648000.0 / np.pi
c = 2.99792458e10

nu0 = 2.466e15
lambda0 = c / nu0 / angstroms

# Figure out how many files we have to work with
colt_files = sorted(glob.glob("rank_[0-9][0-9][0-9]_tot_SMC.h5"))

# Load simulation attributes
with h5py.File(colt_files[0], 'r') as f:
    locals().update(f.attrs)
    locals().update(f['LOS'].attrs)
    Flux_LOS = f['LOS/Flux'][:]
    SB_LOS = f['LOS/SB'][:]
    SB_LOS_error = f['LOS/SB_error'][:]

if freq_type == b'Delta_v':
    Dv_res = d_freq
    l_res = Dv_res * km * lambda0 / c
else:
    raise RuntimeError("unknown/unimplemented frequency type {}".format(freq_type))

# This is our imitation of the MPI reduction
for colt_name in colt_files[1:]:
    with h5py.File(colt_name, 'r') as f:
        Flux_LOS += f['LOS/Flux'][:]
        SB_LOS += f['LOS/SB'][:]
        SB_LOS_error += f['LOS/SB_error'][:]

# compute conversion factors to physical units; this code taken straight from COLT
SB_ipw = arcsec * (1 + z)**2 / (np.sqrt(SB_arcsec2) * d_L)

Flux_LOS_fac = L_Lya / (2 * np.pi * l_res * (1.0 + z) * d_L**2)
SB_dA = arcsec * (1 + z)**2 / SB_ipw
SB_LOS_fac = L_Lya / (2*np.pi * SB_dA**2)
SB_cube_fac = SB_LOS_fac / (l_res * (1 + z))

# Apply conversions
SB_LOS_error = np.sqrt(SB_LOS_error) / SB_LOS
Flux_LOS *= Flux_LOS_fac
SB_LOS *= SB_LOS_fac

print('L_Lya', L_Lya)
stitched_filename = 'stitched_' + colt_files[0][9:]
shutil.copyfile(colt_files[0], stitched_filename)
with h5py.File(stitched_filename, 'r+') as outfile:
    outfile['LOS/Flux'][:] = Flux_LOS
    outfile['LOS/Flux_avg'][:] = np.average(Flux_LOS, axis=0)
    outfile['LOS/SB'][:] = SB_LOS
    outfile['LOS/SB_error'][:] = SB_LOS_error

