import argparse
import numpy as np
import os
import matplotlib
import h5py
from matplotlib import pyplot as plt
import matplotlib.patheffects

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser("COLT Plotter")
parser.add_argument("colt_file")
parser.add_argument("-o", "--output")
parser.add_argument("--dpi", default=96, type=int)
parser.add_argument("--uv", action="store_true", default=False)
parser.add_argument("--muse", action="store_true", default=False)
parser.add_argument("--error", action="store_true", default=False)
parser.add_argument("--sb_image", default=0, type=int)
args = parser.parse_args()

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

if args.error:
    max_clip = 1.0
    min_clip = 0.0
elif args.uv:
    max_clip = 1e-4
    min_clip = 1e-9
else:
    max_clip = 5e-16
    min_clip = 1e-20

bottom = 0.07
top = 0.02
left = 0.08
right = 0.01


with h5py.File(args.colt_file, "r") as f:
    if args.error:
        image = f["LOS/SB_error"][args.sb_image]
        image[np.isnan(image)] = 1.0
    else:
        image = f["LOS/SB"][args.sb_image]

    redshift = f.attrs["z"]
    lum = f.attrs["L_Lya"] * np.sum(f["esc/weight"])
    colt_pixel_size = np.sqrt(f["LOS"].attrs["SB_arcsec2"])
    image_radius = (colt_pixel_size * image.shape[0])/2


fig = plt.figure(figsize=(6.0, 5.0))
ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
ax.text(
    0.5,
    0.99,
    f"z = {redshift:.2f}\nL_Lya = {lum:.2e}",
    horizontalalignment="center",
    verticalalignment="top",
    transform=ax.transAxes,
    color="k",
    path_effects=[matplotlib.patheffects.withStroke(linewidth=5, foreground="w")],
)


if args.error:
    plot_image = ax.imshow(
        image,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1.0),
        extent=[-image_radius, image_radius, -image_radius, image_radius],
        resample=False,
        interpolation="nearest",
    )
    ax.set_xlabel("(arcsec)")
    ax.set_ylabel("(arcsec)")
else:
    if args.muse:
        from scipy.ndimage import zoom
        from scipy.ndimage.filters import gaussian_filter

        muse_pixel_size = 0.2  # arcseconds/pixel
        muse_psf_fwhm = 0.6

        # Apply MUSE resolution via gaussian convolution
        sigma = muse_psf_fwhm / colt_pixel_size / 2.355  # convert from FWHM to sigma
        image = gaussian_filter(image, sigma)

        # Bin to MUSE pixel scale
        #muse_image = zoom(muse_image, colt_pixel_size / muse_pixel_size)
        #print(muse_image.shape)

        #print("colt", np.sum(image) * colt_pixel_size ** 2)
        #print("muse", np.sum(muse_image) * muse_pixel_size ** 2)

    plot_image = ax.imshow(
        image,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
        extent=[-image_radius, image_radius, -image_radius, image_radius],
        resample=False,
        interpolation="nearest",
    )

    '''
    x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    x = x[::-1]
    x = x + 0.5
    y = y + 0.5
    y = (y * image_radius / image.shape[0] * 2) - image_radius
    x = (x * image_radius / image.shape[0] * 2) - image_radius
    ax.contour(y, x, image, levels=[1e-18], colors='w')
    '''

    ax.set_xlabel("(arcsec)")
    ax.set_ylabel("(arcsec)")

cbar = fig.colorbar(plot_image, pad=0)
cbar.ax.tick_params(axis="both", direction="in", which="both")

if args.error:
    cbar.set_label("Relative Error")
elif args.uv:
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
