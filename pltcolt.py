import glob
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py

parser = argparse.ArgumentParser("COLT Plotter")
parser.add_argument("file_patterns", nargs="+")
parser.add_argument("-o", "--output")
args = parser.parse_args()

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

max_clip = 1e-14
min_clip = 1e-23

bottom = 0.09
top = 0.1
left = 0.05
right = 0.05

with PdfPages(args.output) as pdf:
    for pattern in args.file_patterns:
        for output_path in sorted(glob.glob(pattern)):

            with h5py.File(output_path, "r") as f:
                image = f["LOS/SB"][0]
                redshift = f.attrs["z"]
                # COLT units are in cm, convert to kpc
                image_width = f["LOS"].attrs["SB_radius"] / 3.085677581467192e21

            fig = plt.figure(figsize=(12.80, 10.24))

            image = image.clip(min_clip, max_clip)

            fig.suptitle(f"z = {redshift:.2f}", fontsize=48)
            ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
            plot_image = ax.imshow(
                image,
                cmap="magma",
                norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
                extent=[
                    -image_width / 2,
                    image_width / 2,
                    -image_width / 2,
                    image_width / 2,
                ],
                interpolation="gaussian",
                resample=True,
            )
            contour = ax.contour(
                np.flipud(image),
                levels=[1.4e-18],
                colors="w",
                extent=[
                    -image_width / 2,
                    image_width / 2,
                    -image_width / 2,
                    image_width / 2,
                ],
                linewidths=[0.5],
            )
            cbar = fig.colorbar(plot_image, pad=0)
            cbar.ax.tick_params(axis="both", direction="in", which="both")

            ax.set_xlabel("(kpc)", color="w")
            ax.set_ylabel("(kpc)", color="w")
            cbar.set_label(
                "Ly$\\alpha$ Surface Brightness $\left(\\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\\right)$"
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

            pdf.savefig(fig, dpi=100)
            plt.close(fig)
