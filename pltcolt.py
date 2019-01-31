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
parser.add_argument("--dpi", default=96*8, type=int)
parser.add_argument("--uv", action='store_true', default=False)
args = parser.parse_args()

'''
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
'''

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

max_clip = 1e-16
min_clip = 1e-22

bottom = 0.07
top = 0.02
left = 0.08
right = 0.01

with PdfPages(args.output) as pdf:
    for pattern in args.file_patterns:
        for output_path in sorted(glob.glob(pattern)):

            with h5py.File(output_path, "r") as f:
                if args.uv:
                    image = f["LOS/SB_UV_LOS"][0]
                else:
                    image = f["LOS/SB"][0]
                redshift = f.attrs["z"]
                # COLT units are in cm, convert to kpc
                image_radius = f["LOS"].attrs["SB_radius"] / 3.085677581467192e21
                lum = f.attrs["L_Lya"] * np.sum(f['esc/weight'])

            #fig = plt.figure(figsize=(12., 10.))
            fig = plt.figure(figsize=(6.0, 5.0))

            image = image.clip(min_clip, max_clip)
            #fig.suptitle(f"L_Lya = {lum:.2e}", fontsize=36, color='w')
            ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
            ax.text(0.5, 0.99, f"z = {redshift:.2f}\nL_Lya = {lum:.2e}", horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, color='w')
            plot_image = ax.imshow(
                image,
                cmap="viridis",
                norm=matplotlib.colors.LogNorm(vmin=min_clip, vmax=max_clip),
                extent=[
                    -image_radius,
                    image_radius,
                    -image_radius,
                    image_radius,
                ],
                resample=True,
                interpolation="nearest",
            )
            '''
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
            '''
            cbar = fig.colorbar(plot_image, pad=0)
            cbar.ax.tick_params(axis="both", direction="in", which="both")

            ax.set_xlabel("(kpc)")
            ax.set_ylabel("(kpc)")
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

            fig.set_size_inches(12.80, 10.24)
            pdf.savefig(fig, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
