import caesar

caesar.driver.drive(
    snapdirs=["/ufrc/narayanan/kimockb/FIRE2/h113_HR_sn1dy300ro100ss"],
    snapname="snapshot_",
    snapnums=list(range(179, 19, -1)),
    extension="0.hdf5",
    skipran=True,
    progen=True,
)
