import matplotlib

matplotlib.use("Agg")
matplotlib.rc("font", family="STIXGeneral")
# matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({"font.size": 22})
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import numpy as np
import yt
import caesar
from scipy.misc import imread


style_name = {"mathtext.fontset": "cm", "mathtext.fallback_to_cm": True}


with PdfPages("symposium.pdf") as pdf, matplotlib.style.context(
    style_name, after_reset=False
) as cm:

    # Presentation setup
    fig = plt.figure(figsize=(12.80, 10.24))

    bottom = 0.09
    top = 0.1
    left = 0.05
    right = 0.05

    # Title slide
    ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Head in the (HII) Clouds:\nTrying to simulate Lyman-Alpha Blobs with COLT",
        va="center",
        ha="center",
        fontsize=52,
        wrap=True,
    )
    pdf.savefig(fig, dpi=100)
    ax.remove()

    # Quote slide
    ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        "...We have also discovered two extremely bright, large, and diffuse Lyα-emitting “blobs”...",
        va="center",
        ha="left",
        fontsize=52,
        wrap=True,
    )
    text = fig.text(
        1.0, 0.0, "Steidel et al. 2000", va="bottom", ha="right", fontsize=20
    )
    pdf.savefig(fig, dpi=100)
    text.remove()
    ax.remove()

    # Image slide
    fig.suptitle("The First Blob", fontsize=52)
    ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
    ax.axis("off")

    image = ax.imshow(imread("steidel_2000.gif"), cmap="Greys_r")
    text = fig.text(
        1.0, 0.0, "Steidel et al. 2000", va="bottom", ha="right", fontsize=20
    )
    pdf.savefig(fig, dpi=100)
    text.remove()
    ax.remove()

    # Bullet-ish slide
    fig.suptitle("Lyman-alpha", fontsize=52)
    ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
    ax.axis("off")
    text = fig.text(
        0.05,
        0.5,
        "UV, redshifted to optical for 2 < z < 5\n\nVery fast transition\n\nVery optically thick",
        va="center",
        ha="left",
        fontsize=48,
        wrap=True,
    )
    pdf.savefig(fig, dpi=100)
    text.remove()
    ax.remove()

    ds = yt.load('/ufrc/narayanan/desika.narayanan/gizmo_runs/MassiveFIRE/FIRE2/A2_res33000/snapdir_120/snapshot_120.0.hdf5')
    obj = caesar.load('/ufrc/narayanan/kimockb/fire_caesar/caesar_snapshot_120.hdf5')
    
    # Plot our ProjectionPlots with their own axes objects
    for axis in ['x', 'y', 'z']:
        plot = yt.ProjectionPlot(
            ds,
            axis=axis,
            fields=("gas", "density"),
            width=(250, "kpc"),
            center=obj.galaxies[0].pos.in_units("kpc"),
        )
        plot.set_font_size(22)
        plot.set_cmap(("gas", "density"), cmap="inferno") # TODO: Why doesn't this do anything
        density_plot = plot.plots[("gas", "density")]
        density_plot.figure.set_size_inches(12.80, 10.24)
        density_plot.axes.set_position([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        density_plot.figure.suptitle("Is this an LAB?", fontsize=52)

        # God dammit put the colorbar where I want it
        density_plot.cax.set_position([0.815, 0.09, (0.9-0.09)/25, 0.9-0.09])
        pdf.savefig(density_plot.figure, dpi=100)

    # Figure slide
    colt_file =  h5py.File("/ufrc/narayanan/kimockb/fireA2/120/tot_SMC.h5")
    colt_images = colt_file["LOS/SB"][:4]
    for image in colt_images:
        image = image.clip(1e-23, 1e-10)

        fig.suptitle(f"Is this an LAB? (z={colt_file.attrs['z']:.2f})", fontsize=52)

        ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        plot_image = ax.imshow(
            np.flipud(image),
            cmap="inferno",
            norm=LogNorm(vmin=1e-23, vmax=1e-10, clip=True),
            extent=[-250, 250, -250, 250],
        )
        contour = ax.contour(image, levels=[1.4e-18], colors='w', extent=[-250, 250, -250, 250], linewidths=[0.5])
        cbar = fig.colorbar(plot_image, pad=0)

        ax.set_xlabel("(kpc)")
        ax.set_ylabel("(kpc)")
        cbar.set_label(
            "Ly$\\alpha$ Surface Brightness $\left(\\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\\right)$"
        )

        pdf.savefig(fig, dpi=100)
        cbar.remove()
        ax.remove()

    ds = yt.load('/ufrc/narayanan/desika.narayanan/gizmo_runs/MassiveFIRE/FIRE2/A2_res33000/snapdir_172/snapshot_172.0.hdf5')
    obj = caesar.load('/ufrc/narayanan/kimockb/fire_caesar/caesar_snapshot_172.hdf5')
    
    # Plot our ProjectionPlots with their own axes objects
    for axis in ['x', 'y', 'z']:
        plot = yt.ProjectionPlot(
            ds,
            axis=axis,
            fields=("gas", "density"),
            width=(250, "kpc"),
            center=obj.galaxies[0].pos.in_units("kpc"),
        )
        plot.set_font_size(22)
        plot.set_cmap(("gas", "density"), cmap="inferno") # TODO: Why doesn't this do anything
        density_plot = plot.plots[("gas", "density")]
        density_plot.figure.set_size_inches(12.80, 10.24)
        density_plot.axes.set_position([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        density_plot.figure.suptitle("Is this an LAB?", fontsize=52)

        # God dammit put the colorbar where I want it
        density_plot.cax.set_position([0.815, 0.09, (0.9-0.09)/25, 0.9-0.09])
        pdf.savefig(density_plot.figure, dpi=100)

    # Figure slide
    colt_file =  h5py.File("/ufrc/narayanan/kimockb/fireA2/172/tot_SMC.h5")
    colt_images = colt_file["LOS/SB"][:4]
    for image in colt_images:
        image = image.clip(1e-23, 1e-10)

        fig.suptitle(f"Is this an LAB? (z={colt_file.attrs['z']:.2f})", fontsize=52)

        ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        plot_image = ax.imshow(
            np.flipud(image),
            cmap="inferno",
            norm=LogNorm(vmin=1e-23, vmax=1e-10, clip=True),
            extent=[-250, 250, -250, 250],
        )
        contour = ax.contour(image, levels=[1.4e-18], colors='w', extent=[-250, 250, -250, 250], linewidths=[0.5])
        cbar = fig.colorbar(plot_image, pad=0)

        ax.set_xlabel("(kpc)")
        ax.set_ylabel("(kpc)")
        cbar.set_label(
            "Ly$\\alpha$ Surface Brightness $\left(\\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\\right)$"
        )

        pdf.savefig(fig, dpi=100)
        cbar.remove()
        ax.remove()

    ds = yt.load('/ufrc/narayanan/desika.narayanan/gizmo_runs/MassiveFIRE/FIRE2/A2_res33000/snapdir_214/snapshot_214.0.hdf5')
    obj = caesar.load('/ufrc/narayanan/kimockb/fire_caesar/caesar_snapshot_214.hdf5')
    
    # Plot our ProjectionPlots with their own axes objects
    for axis in ['x', 'y', 'z']:
        plot = yt.ProjectionPlot(
            ds,
            axis=axis,
            fields=("gas", "density"),
            width=(250, "kpc"),
            center=obj.galaxies[0].pos.in_units("kpc"),
        )
        plot.set_font_size(22)
        plot.set_cmap(("gas", "density"), cmap="inferno") # TODO: Why doesn't this do anything
        density_plot = plot.plots[("gas", "density")]
        density_plot.figure.set_size_inches(12.80, 10.24)
        density_plot.axes.set_position([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        density_plot.figure.suptitle("Is this an LAB?", fontsize=52)

        # God dammit put the colorbar where I want it
        density_plot.cax.set_position([0.815, 0.09, (0.9-0.09)/25, 0.9-0.09])
        pdf.savefig(density_plot.figure, dpi=100)

    # Figure slide
    colt_file =  h5py.File("/ufrc/narayanan/kimockb/fireA2/214/tot_SMC.h5")
    colt_images = colt_file["LOS/SB"][:4]
    for image in colt_images:
        image = image.clip(1e-23, 1e-10)

        fig.suptitle(f"Is this an LAB? (z={colt_file.attrs['z']:.2f})", fontsize=52)

        ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        plot_image = ax.imshow(
            np.flipud(image),
            cmap="inferno",
            norm=LogNorm(vmin=1e-23, vmax=1e-10, clip=True),
            extent=[-250, 250, -250, 250],
        )
        contour = ax.contour(image, levels=[1.4e-18], colors='w', extent=[-250, 250, -250, 250], linewidths=[0.5])
        cbar = fig.colorbar(plot_image, pad=0)

        ax.set_xlabel("(kpc)")
        ax.set_ylabel("(kpc)")
        cbar.set_label(
            "Ly$\\alpha$ Surface Brightness $\left(\\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\\right)$"
        )

        pdf.savefig(fig, dpi=100)
        cbar.remove()
        ax.remove()

    ds = yt.load('/ufrc/narayanan/desika.narayanan/gizmo_runs/MassiveFIRE/FIRE2/A2_res33000/snapdir_277/snapshot_277.0.hdf5')
    obj = caesar.load('/ufrc/narayanan/kimockb/fire_caesar/caesar_snapshot_277.hdf5')
    
    # Plot our ProjectionPlots with their own axes objects
    for axis in ['x', 'y', 'z']:
        plot = yt.ProjectionPlot(
            ds,
            axis=axis,
            fields=("gas", "density"),
            width=(250, "kpc"),
            center=obj.galaxies[0].pos.in_units("kpc"),
        )
        plot.set_font_size(22)
        plot.set_cmap(("gas", "density"), cmap="inferno") # TODO: Why doesn't this do anything
        density_plot = plot.plots[("gas", "density")]
        density_plot.figure.set_size_inches(12.80, 10.24)
        density_plot.axes.set_position([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        density_plot.figure.suptitle("Is this an LAB?", fontsize=52)

        # God dammit put the colorbar where I want it
        density_plot.cax.set_position([0.815, 0.09, (0.9-0.09)/25, 0.9-0.09])
        pdf.savefig(density_plot.figure, dpi=100)

    # Figure slide
    colt_file =  h5py.File("/ufrc/narayanan/kimockb/fireA2/277/tot_SMC.h5")
    colt_images = colt_file["LOS/SB"][-4:]
    for image in colt_images:
        image = image.clip(1e-23, 1e-10)

        fig.suptitle(f"Is this an LAB? (z={colt_file.attrs['z']:.2f})", fontsize=52)

        ax = fig.add_axes([left, bottom, 1.0 - left - right, 1.0 - bottom - top])
        plot_image = ax.imshow(
            np.flipud(image),
            cmap="inferno",
            norm=LogNorm(vmin=1e-23, vmax=1e-10, clip=True),
            extent=[-250, 250, -250, 250],
        )
        contour = ax.contour(image, levels=[1.4e-18], colors='w', extent=[-250, 250, -250, 250], linewidths=[0.5])
        cbar = fig.colorbar(plot_image, pad=0)

        ax.set_xlabel("(kpc)")
        ax.set_ylabel("(kpc)")
        cbar.set_label(
            "Ly$\\alpha$ Surface Brightness $\left(\\frac{\mathrm{erg}}{\mathrm{s}\,\mathrm{cm}^2\,\mathrm{arcsec}^2}\\right)$"
        )

        pdf.savefig(fig, dpi=100)
        cbar.remove()
        ax.remove()

