import argparse
import numpy as np
import matplotlib
import h5py
from matplotlib import pyplot as plt
import matplotlib.patheffects
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erf
import numba


# Constants
n_photons = 1e8
Δμ = 0.0015957691216  # Gaussian model: 10^7 photons, ~1% Poisson error
μ_min_cut = max(0.0, 1.0 - 6.0 * Δμ)
n_photons_per_pixel = (float(n_photons) * 0.5 * np.sqrt(np.pi / 2.0) * Δμ *
                       erf(1.0 / (np.sqrt(2.0) * Δμ)))

@numba.njit
def f_esc_at_angle(k, k_esc, weights):

    f_esc = 0.0
    for i in range(k_esc.shape[0]):
        distance = k[0] * k_esc[i, 0] + k[1] * k_esc[i, 1] + k[2] * k_esc[i, 2]

        if distance > μ_min_cut:
            los_weight = (np.exp(-(1.0 - distance)**2 / (2.0 * Δμ**2)) /
                          n_photons_per_pixel)

            f_esc += los_weight * (weights[i] * n_photons)

    return f_esc


matplotlib.rcParams['font.size'] = 20

parser = argparse.ArgumentParser("COLT Plotter")
parser.add_argument("colt_file")
parser.add_argument("-o", "--output")
parser.add_argument("--max", default=5e-16, type=float)
parser.add_argument("--min", default=1e-19, type=float)
args = parser.parse_args()


with h5py.File(args.colt_file, "r") as f:
    los_images = f["LOS/SB"][:]

    redshift = f.attrs["z"]
    lum = f.attrs["L_Lya"]
    colt_pixel_size = np.sqrt(f["LOS"].attrs["SB_arcsec2"])
    image_radius = f["LOS"].attrs["SB_radius"] / (3.085_677_581_467_192e18 * 1e3)
    direction = f['esc/direction'][:]
    weight = f['esc/weight'][:]
    los_angles = f['LOS']['k'][:]

with PdfPages(args.output) as pdf:
    for (angle, image) in zip(los_angles, los_images):

        escape = f_esc_at_angle(angle, direction, weight)
        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.gca()
        ax.text(
            0.5,
            0.99,
            f"$z = {redshift:.2f}$\n$L_{{Ly\\alpha}} = ${(lum*escape):.2e}",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            color="w",
            path_effects=[
                matplotlib.patheffects.withStroke(
                    linewidth=6, foreground=matplotlib.cm.viridis(0)
                )
            ],
        )

        plot_image = ax.imshow(
            np.clip(image, args.min, args.max),
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(vmin=args.min, vmax=args.max),
            extent=[-image_radius, image_radius, -image_radius, image_radius],
            resample=False,
            interpolation="none",
        )

        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")

        l, b, w, h = ax.get_position().bounds
        cax = fig.add_axes([l + w, b, 0.04, h])
        cax.set_snap(True)
        cbar = fig.colorbar(plot_image, pad=0, cax=cax)
        cbar.ax.tick_params(axis="both", direction="in", which="both")

        cbar.set_label(
            r"$\Sigma_{\rm Ly \alpha} \left(\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{arcsec}^{-2}\right)$"
        )

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))

        pdf.savefig(fig, bbox_inches='tight')
