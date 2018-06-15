import numpy as np
import loadtxt
import time

def ahf_to_caesar(filename, num_particles):
    ahf_particles = loadtxt.loadtxt(filename, skiprows=1)
    # ahf_particles should be (particle_id, particle_type)

    table = []
    halo_start = 0
    i = 0
    while i < ahf_particles.shape[0]:
        halo_len, halo_num = int(ahf_particles[i][0]), int(ahf_particles[i][1])
        # print halo_len, halo_num
        for particle in ahf_particles[i+1: i+halo_len+1]:
            table.append((int(particle[0]), int(particle[1]), int(halo_num)))

        i += halo_len + 1

    # table is (particle_id, particle_type, halo_num)
    table = list(sorted(table, key=lambda row: row[0]))

    output = []
    particle_id = 0
    row = 0
    for particle_id in range(num_particles):
        if table[row][0] == particle_id:
            output.append(table[row][2])
            row += 1
        else:
            output.append(-1)
    
    return np.array(output)

