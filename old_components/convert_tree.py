import numpy as np
import octree
import h5py
import fire


def convert_tree(octree_filename, output_filename, ionization_filename):
    # load tree
    tree = octree.TREE(octree_filename).load()

    # load ionization data
    state = np.loadtxt(ionization_filename)
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
    f = h5py.File(output_filename, 'w')
    f.attrs['redshift'] = 1.0
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
    # hl = sp.loadhalo(id=id)
    # p4 = sp.loadpart(4)
    # r4 = np.sqrt((p4.p[:,0]-hl.xc)**2+(p4.p[:,1]-hl.yc)**2+(p4.p[:,2]-hl.zc)**2)
    # p4.p[:,0] -= hl.xc; p4.p[:,1] -= hl.yc; p4.p[:,2] -= hl.zc # shift coordinates
    # ok = (r4<fvir*hl.rvir)
    # f.attrs["Redshift"] = sp.redshift
    # f.create_dataset("star_vel", data=p4.v[ok]) # in km/s
    # f.create_dataset("star_pos", data=p4.p[ok])
    # f.create_dataset("star_m", data=p4.m[ok]) # in 10^10 Msun
    # f.create_dataset("star_age", data=p4.age[ok]) # in Gyr
    # f.create_dataset("star_z", data=p4.z[ok]) # in mass fraction


if __name__ == '__main__':
    fire.Fire(convert_tree)
