import sys
import os
import numpy as np
from snap2gadget import snap2gadget
import IPython
import yt
import numba

AHF_TEMPLATE = """
[AHF]

# (stem of the) filename from which to read the data to be analysed
ic_filename = {}

# what type of input file (cf. src/libio/io_file.h)
ic_filetype       = 60

# prefix for the output files
outfile_prefix     = {}

# number of grid cells for the domain grid (1D)
LgridDomain       = 64

# number of grid cells for the domain grid (1D) (limits spatial resolution to BoxSize/LgridMax)
LgridMax          = 16777216 

# refinement criterion on domain grid (#particles/cell)
NperDomCell       = 2.0

# refinement criterion on all higher resolution grids (#particles/cells)
NperRefCell       = 2.5

# particles with velocity v > VescTune x Vesc are considered unbound 
VescTune          = 1.5 

# minimum number of particles for a halo
NminPerHalo       = 20

# normalisation for densities (1: RhoBack(z), 0:RhoCrit(z))
RhoVir            = 0

# virial overdensity criterion (<0: let AHF calculate it); Rvir is defined via M(<Rvir)/Vol = Dvir * RhoVir
Dvir              = 200 

# maximum radius (in Mpc/h) used when gathering initial set of particles for each halo (should be larger than the largest halo expected)
MaxGatherRad      = 3.0	

# the level on which to perform the domain decomposition (MPI only, 4=16^3, 5=32^3, 6=64^3, 7=128^3, 8=256^3, etc.)
LevelDomainDecomp = 6

# how many CPU's for reading (MPI only)
NcpuReading       = 4

# name of file containing the dark energy relevant tables (only relevant for -DDARK_ENERGY)
de_filename       = my_dark_energy_table.txt


############################### FILE SPECIFIC DEFINITIONS ###############################

# NOTE: all these factors are supposed to transform your internal units to
#           [x] = Mpc/h
#           [v] = km/sec
#           [m] = Msun/h
#           [e] = (km/sec)^2

[GADGET]
GADGET_LUNIT      = 1e-3
GADGET_MUNIT      = 1e10

[TIPSY]
TIPSY_BOXSIZE       = 50.0
TIPSY_MUNIT         = 4.75e16
TIPSY_VUNIT         = 1810.1
TIPSY_EUNIT         = 0.0
TIPSY_OMEGA0        = 0.24
TIPSY_LAMBDA0       = 0.76

[ART]
ART_BOXSIZE         = 20
ART_MUNIT           = 6.5e8
"""


class Brutus:
    def __init__(self, filename):
        path = os.path.abspath(filename)
        self.path = path
        self.halos = []

        # Convert into the stupid gadget binary format
        name = path.rsplit('.', 1)[0]
        gadget_path = name + '.gadget2'
        config_path = name + '.AHF'
        snap2gadget(path, gadget_path)

        # Create an AHF config file
        with open(config_path, 'w') as ahf_config_file:
            ahf_config_file.write(AHF_TEMPLATE.format(gadget_path, name))
    
        # TODO: Uncomment this
        #os.system('AHF-v1.0-094 ' + config_name)

        # Locate and open the AHF halos file
        potential_particle_files = [f.path for f in os.scandir(os.path.dirname(path)) if f.name.startswith(os.path.basename(name)) and f.name.endswith('AHF_particles')]
        
        if len(potential_particle_files) != 1:
            print(potential_particle_files)
            raise RuntimeError("Cannot find the right particle file")
        
        particles_file = potential_particle_files[0]

        ahf_particles = np.loadtxt(particles_file, dtype=int, skiprows=1)
        # ahf_particles should be (particle_id, particle_type)

        ds = yt.load(filename)
        data = ds.all_data()

        i = 0
        while i < ahf_particles.shape[0]:
            halo_len, halo_num = int(
                ahf_particles[i][0]), int(
                ahf_particles[i][1])
            current_halo = ahf_particles[i + 1: i + halo_len + 1]

            gas_ids = current_halo[:, 0][current_halo[:, 1] == 0]
            dm_ids = current_halo[:, 0][current_halo[:, 1] == 1]
            star_ids = current_halo[:, 0][current_halo[:, 1] == 4]

            glist = id_to_index(gas_ids, np.array(data[('PartType0', 'ParticleIDs')]))
            slist = id_to_index(star_ids, np.array(data[('PartType4', 'ParticleIDs')]))
            dmlist = id_to_index(dm_ids, np.array(data[('PartType1', 'ParticleIDs')]))

            self.halos.append(Halo(glist=glist, slist=slist, dmlist=dmlist))

            i += halo_len + 1


@numba.jit(nopython=True, cache=True)
def id_to_index(parttype_ids, all_ids):
    output = np.empty_like(parttype_ids)

    for p in range(len(parttype_ids)):
        for i in range(len(all_ids)):
            if all_ids[i] == parttype_ids[p]:
                output[p] = i
                break

    return output

class Halo:
    def __init__(self, *, dmlist, slist, glist):
        self.slist = slist
        self.dmlist = dmlist
        self.glist = glist

if __name__ == '__main__':
    print(sys.argv[1])
    obj = Brutus(sys.argv[1])
    IPython.embed()


