import argparse
import numpy as np
import os
import matplotlib
import yt
import h5py
from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser("COLT Plotter")
parser.add_argument("ds")
parser.add_argument("colt_file")
parser.add_argument("-o", "--output")
parser.add_argument("--dpi", default=96, type=int)
parser.add_argument("--uv", action="store_true", default=False)
parser.add_argument("--muse", action="store_true", default=False)
parser.add_argument("--sb_image", default=0, type=int)
args = parser.parse_args()

"""
matplotlib.rcParams.update(
    {
        "savefig.facecolor": "k",
        "text.color": "w",
        "axes.edgecolor": "w",
        "axes.labelcolor": "w",
        "xtick.color": "w",
        "ytick.color": "w",
        "font.family": "STIXGeneral",
        "font.size": 22,
        "mathtext.fontset": "cm",
        "mathtext.fallback_to_cm": True,
    }
)
"""

matplotlib.rcParams.update(
    {
        "savefig.facecolor": "w",
        "text.color": "k",
        "axes.edgecolor": "k",
        "axes.labelcolor": "k",
        "xtick.color": "k",
        "ytick.color": "k",
        "font.family": "STIXGeneral",
        "font.size": 22,
        "mathtext.fontset": "cm",
        "mathtext.fallback_to_cm": True,
    }
)

if args.uv:
    max_clip = 1e-4
    min_clip = 1e-9
else:
    max_clip = 1e-16
    min_clip = 1e-22

bottom = 0.07
top = 0.02
left = 0.08
right = 0.01

ds = yt.load(args.ds)

with h5py.File(args.colt_file, "r") as f:
    image = f["LOS/SB"][args.sb_image]
    redshift = f.attrs["z"]
    # COLT units are in cm, convert to kpc
    lum = f.attrs["L_Lya"] * np.sum(f["esc/weight"])
    print('luminosity', lum)
    image_radius = ds.quan(250.0, 'kpccm')
    #image_radius = ds.quan(f["LOS"].attrs["SB_radius"], "cm").to("kpccm")
    colt_pixel_size = np.sqrt(f["LOS"].attrs["SB_arcsec2"])


#image = image.clip(min_clip, max_clip)

fig = plt.figure(figsize=(6.0, 5.0))
ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
ax.text(
    0.5,
    0.99,
    f"z = {redshift:.2f}\nL_Lya = {lum:.2e}",
    horizontalalignment="center",
    verticalalignment="top",
    transform=ax.transAxes,
    color="w",
)

if args.muse:
    from scipy.ndimage import zoom
    from scipy.ndimage.filters import gaussian_filter

    muse_pixel_size = 0.2 # arcseconds/pixel
    muse_psf_fwhm = 0.6

    # Apply MUSE resolution via gaussian convolution
    sigma = muse_psf_fwhm / colt_pixel_size / 2.355 # convert from FWHM to sigma
    muse_image = gaussian_filter(image, sigma)
   
    # Bin to MUSE pixel scale
    muse_image = zoom(muse_image, colt_pixel_size / muse_pixel_size)

    print('colt', np.sum(image) * colt_pixel_size**2)
    print('muse', np.sum(muse_image) * muse_pixel_size**2)

    image_radius /= ds.cosmology.angular_scale(0.0, ds.current_redshift).to("kpccm/arcsec")

    plot_image = ax.imshow(
        muse_image,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
        extent=[-image_radius, image_radius, -image_radius, image_radius],
        resample=False,
        interpolation="nearest",
    )
    ax.set_xlabel("(arcsec)")
    ax.set_ylabel("(arcsec)")

else:
    plot_image = ax.imshow(
        image,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
        extent=[-image_radius, image_radius, -image_radius, image_radius],
        resample=False,
        interpolation="nearest",
    )
    ax.set_xlabel("(comoving kpc)")
    ax.set_ylabel("(comoving kpc)")

cbar = fig.colorbar(plot_image, pad=0)
cbar.ax.tick_params(axis="both", direction="in", which="both")

if args.uv:
    cbar.set_label(
        r"Lyman-Continuum Surface Brightness $\left(\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\right)$"
    )
else:
    cbar.set_label(
        r"Ly$\alpha$ Surface Brightness $\left(\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\right)$"
    )
ax.tick_params(
    axis="both",
    direction="in",
    which="both",
    bottom=True,
    top=True,
    left=True,
    right=True,
)
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

fig.set_size_inches(12.80, 10.24)
fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
