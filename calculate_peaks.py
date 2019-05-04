import numpy as np
import h5py
from scipy.special import erf

plot_dir = '../../plots/_src_'
import sys; sys.path.insert(0, plot_dir)
import cosmology as cosmo
from constants import *

def v_peak_FWHM(X, Y):
    imax = np.argmax(Y)
    x1, x2, x3 = X[imax-1], X[imax], X[imax+1] # Fit a parabola locally
    y1, y2, y3 = Y[imax-1], Y[imax], Y[imax+1] # to find a smoother max
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    pA = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    pB = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
    pC = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xv = -pB / (2.*pA)
    yv = pC - pB**2 / (4.*pA)
    #assert yv >= Y[imax]
    #if Y[imax] > yv:
    #    print('\nWarning: Y[imax] =',Y[imax],'and yv =',yv,'        X[imax] =',X[imax],'and xv =',xv)
    Yhalf = 0.5 * yv
    iFWHMp = imax
    while Y[iFWHMp] > Yhalf:
        iFWHMp += 1
    xFWHMp = (X[iFWHMp]-X[iFWHMp-1]) * (Yhalf-Y[iFWHMp-1]) / (Y[iFWHMp]-Y[iFWHMp-1]) + X[iFWHMp-1]
    iFWHMm = imax
    while Y[iFWHMm] > Yhalf:
        iFWHMm -= 1
    xFWHMm = (X[iFWHMm]-X[iFWHMm+1]) * (Yhalf-Y[iFWHMm+1]) / (Y[iFWHMm]-Y[iFWHMm+1]) + X[iFWHMm+1]
    assert xFWHMp > xFWHMm
    return xv, xFWHMp - xFWHMm

def weighted_median(Z, W):
    # Z = data, W = weights
    isort = np.argsort(Z)
    Z_sorted = Z[isort]
    W_sorted = W[isort]
    IW_sorted = np.cumsum(W_sorted)
    IW_sorted /= IW_sorted[-1]
    i = 0
    while IW_sorted[i] <= 0.5:
        i += 1
    return (Z_sorted[i]-Z_sorted[i-1]) * (.5-IW_sorted[i-1]) / (IW_sorted[i]-IW_sorted[i-1]) + Z_sorted[i-1]

def write_snap(snap=67, n_exp=1, emission='rec', v_com=np.zeros(3), verbose=False):
    v_com_mag = np.sqrt(np.sum(v_com**2))
    print('\nsnap =', snap, ' emission =', emission, ' [v_com =', v_com, 'km/s, |v_com| =', v_com_mag, 'km/s]')
    M_UV_filename = 'dust-s{}-r02-bin-SMC_{}.h5'.format(snap,n_exp)
    colt_filename = 'merged_data/HiZFIRE_s{}-r02-bin_{}_SMC.h5'.format(snap,emission)
    fesc_filename = 'peak_s{}_{}_{}.h5'.format(snap,emission,n_exp)
    with h5py.File(M_UV_filename,'r') as df, h5py.File(colt_filename,'r') as cf:
        if verbose:
            print('Opened', M_UV_filename)
            print('File attrs:', [item for item in df.attrs.items()])
            print('File keys: ', [key for key in df.keys()])
            print('Opened', colt_filename)
            print('File attrs:', [item for item in cf.attrs.items()],
                                 [item for item in cf['esc'].attrs.items()])
            print('File keys: ', [key for key in cf['esc'].keys()])
        n_LOS = df.attrs['n_LOS']
        ks = df['k'][:,:]
        n_escaped = cf['esc'].attrs['n_escaped']
        n_photons = cf.attrs['n_photons']
        k_esc = np.vstack([cf['esc']['kx'][:], cf['esc']['ky'][:], cf['esc']['kz'][:]]).T
        r_esc = np.vstack([cf['esc']['x'][:], cf['esc']['y'][:], cf['esc']['z'][:]]).T
        # Distance from a line and a point  = || (r-p) - ((r-p) * k) k ||
        # Line = r + t*k, p = (0,0,0)  =>   = || r - (r*k) k ||
        r_dot_k = r_esc[:,0]*k_esc[:,0] + r_esc[:,1]*k_esc[:,1] + r_esc[:,2]*k_esc[:,2]
        x_impact = r_esc[:,0] - r_dot_k*k_esc[:,0]
        y_impact = r_esc[:,1] - r_dot_k*k_esc[:,1]
        z_impact = r_esc[:,2] - r_dot_k*k_esc[:,2]
        r_impact = np.sqrt(x_impact*x_impact + y_impact*y_impact + z_impact*z_impact)
        del r_dot_k, x_impact, y_impact, z_impact
        freq_esc = cf['esc']['freq'][:]  # Original frequency
        freq_esc += k_esc[:,0]*v_com[0] + k_esc[:,1]*v_com[1] + k_esc[:,2]*v_com[2] # Doppler shifted frequency
        red_mask = (freq_esc >= 0)
        z = cf.attrs['z']
        d_L = cf.attrs['d_L']
        SB_radius = cf['LOS'].attrs['SB_radius']
        #if n_exp == 360:
        #    noise_fac = 0.5
        #else:
        #    noise_fac = 2. # Increases area (therefore photon count) by this factor
        #dA = noise_fac * 4.*np.pi / float(n_LOS)
        #μ_min = np.abs(1. - noise_fac * 2./float(n_LOS))
        #n_photons_per_pixel = noise_fac * float(n_photons) / float(n_LOS)

        # TODO: Corresponds specifically to number of photons; for 1e8 photons
        # we would use
        # Δμ = 0.00015957691216 # Gaussian model: 10^8 photons, ~1% Poisson error
        Δμ = 0.0015957691216 # Gaussian model: 10^7 photons, ~1% Poisson error
        μ_min = 1. - Δμ
        μ_min_cut = np.max([0.,1.-6.*Δμ])
        n_photons_per_pixel = float(n_photons) * 0.5*np.sqrt(np.pi/2.)*Δμ*erf(1./(np.sqrt(2.)*Δμ))
        if verbose:
            print('z =', z)
            print('n_LOS =', n_LOS)
            #print('dA    =', dA)
            print('Δμ    =', Δμ)
            print('μ_min =', μ_min)
            print('θ_min =', 180/np.pi*np.arccos(μ_min), 'degrees')
            #print('noise_fac =', noise_fac)
            print('# photons / pixel =', n_photons_per_pixel)
            print('average accuracy  =', 100./np.sqrt(n_photons_per_pixel), '%\n')
        L_α = cf.attrs['L_Lya'] * Lsun
        L_UV_tot = df.attrs['L_UV_tot']
        M_UV_tot = df.attrs['M_UV_tot']
        EW_α_INT = L_α / L_UV_tot
        L_UV_LOS = df['L_UV_LOS'][:]
        M_UV_LOS = df['M_UV_LOS'][:]
        f_esc_UV_LOS = df['f_esc_UV_LOS'][:]
    N_1_LOS = np.ones_like(f_esc_UV_LOS)
    N_W_LOS = np.ones_like(f_esc_UV_LOS)
    N_W_IGMA_LOS = np.ones_like(f_esc_UV_LOS)
    f_esc_α_LOS = np.ones_like(f_esc_UV_LOS)
    f_esc_α_IGMA_LOS = np.ones_like(f_esc_UV_LOS)
    F_rb_LOS = np.zeros_like(f_esc_UV_LOS)
    F_rb_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    v_avg_LOS = np.zeros_like(f_esc_UV_LOS)
    v_avg_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    v2_avg_LOS = np.zeros_like(f_esc_UV_LOS)
    v2_avg_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    v_std_LOS = np.zeros_like(f_esc_UV_LOS)
    v_std_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    v_peak_LOS = np.zeros_like(f_esc_UV_LOS)
    v_peak_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    FWHM_LOS = np.zeros_like(f_esc_UV_LOS)
    FWHM_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    R_scale_LOS = np.zeros_like(f_esc_UV_LOS)
    R_scale_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    A_scale_LOS = np.zeros_like(f_esc_UV_LOS)
    A_scale_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    R_half_LOS = np.zeros_like(f_esc_UV_LOS)
    R_half_IGMA_LOS = np.zeros_like(f_esc_UV_LOS)
    n_r_bins = 200
    n_logr_bins = 200
    n_Dv_bins = 200
    Flux_LOS = np.zeros([n_LOS,n_Dv_bins])
    IFlux_LOS = np.zeros_like(Flux_LOS)
    Flux_IGMA_LOS = np.zeros_like(Flux_LOS)
    IFlux_IGMA_LOS = np.zeros_like(Flux_LOS)
    SB_r_LOS = np.zeros([n_LOS,n_r_bins])
    ISB_r_LOS = np.zeros_like(SB_r_LOS)
    SB_r_IGMA_LOS = np.zeros_like(SB_r_LOS)
    ISB_r_IGMA_LOS = np.zeros_like(SB_r_LOS)
    SB_logr_LOS = np.zeros([n_LOS,n_logr_bins])
    ISB_logr_LOS = np.zeros_like(SB_logr_LOS)
    SB_logr_IGMA_LOS = np.zeros_like(SB_logr_LOS)
    ISB_logr_IGMA_LOS = np.zeros_like(SB_logr_LOS)
    F_red_r_LOS = np.zeros_like(SB_r_LOS)
    F_blue_r_LOS = np.zeros_like(SB_r_LOS)
    F_red_r_IGMA_LOS = np.zeros_like(SB_r_LOS)
    F_blue_r_IGMA_LOS = np.zeros_like(SB_r_LOS)
    W = np.ones(n_escaped) / float(n_photons)
    W_IGMA = cosmo.F_laursen(freq_esc, z, plot_dir=plot_dir) # TODO: Need stuff from colt-ben/_src_/cosmology.py
    f_esc_α = float(n_escaped) / float(n_photons)
    f_esc_α_IGMA = np.sum(W_IGMA) / float(n_photons)
    T_IGMA = f_esc_α_IGMA / f_esc_α
    F_rb = np.sum(W[red_mask]) / np.sum(W[~red_mask])
    F_rb_IGMA = np.sum(W_IGMA[red_mask]) / np.sum(W_IGMA[~red_mask])
    v_avg = np.mean(freq_esc)
    v2_avg = np.mean(freq_esc**2)
    v_std = np.std(freq_esc)
    v_avg_IGMA = np.average(freq_esc, weights=W_IGMA)
    v2_avg_IGMA = np.average(freq_esc**2, weights=W_IGMA)
    v_std_IGMA = np.sqrt(v2_avg_IGMA - v_avg_IGMA**2)
    ## Flux histogram ##
    # Weight = L_α / (2π n_photons d_L^2 l_res)
    #hist, Δv_edges = np.histogram(freq_esc, bins='auto')
    #Δv_bin = 0.5 * (Δv_edges[1:] + Δv_edges[:-1])
    #dλ = (Δv_edges[1:] - Δv_edges[:-1]) * km * lambda0 * (1.+z) / c
    #Flux = L_α * hist / (2.*np.pi * float(n_photons) * d_L**2 * dλ)
    #IFlux = np.cumsum(Flux*dλ)
    #IFlux /= IFlux[-1]
    #v_peak, FWHM = v_peak_FWHM(Δv_bin, Flux)
    #hist_IGMA, temp = np.histogram(freq_esc, bins=Δv_edges, weights=W_IGMA)
    #Flux_IGMA = L_α * hist_IGMA / (2.*np.pi * float(n_photons) * d_L**2 * dλ)
    #IFlux_IGMA = np.cumsum(Flux_IGMA*dλ)
    #IFlux_IGMA /= IFlux_IGMA[-1]
    #v_peak_IGMA, FWHM_IGMA = v_peak_FWHM(Δv_bin, Flux_IGMA)
    ## SB_r histogram ##
    # Weight = L_α / (2π n_photons arcsec^2 (1+z)^4 dA)
    #r_hist, r_edges = np.histogram(r_impact, bins='auto')
    #dA = 4.*np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
    #SB_r = L_α * r_hist / (2.*np.pi * float(n_photons) * arcsec**2 * (1.+z)**4 * dA)
    #ISB_r = np.cumsum(SB_r*dA)
    #ISB_r /= ISB_r[-1]
    #r_hist_IGMA, temp = np.histogram(r_impact, bins=r_edges, weights=W_IGMA)
    #SB_r_IGMA = L_α * r_hist_IGMA / (2.*np.pi * float(n_photons) * arcsec**2 * (1.+z)**4 * dA)
    #ISB_r_IGMA = np.cumsum(SB_r_IGMA*dA)
    #ISB_r_IGMA /= ISB_r_IGMA[-1]
    ## F_rb as a function of radius ##
    #F_red_r, temp = np.histogram(r_impact[red_mask], bins=r_edges)
    #F_blue_r, temp = np.histogram(r_impact[~red_mask], bins=r_edges)
    #F_red_r_IGMA, temp = np.histogram(r_impact[red_mask], bins=r_edges, weights=W_IGMA[red_mask])
    #F_blue_r_IGMA, temp = np.histogram(r_impact[~red_mask], bins=r_edges, weights=W_IGMA[~red_mask])
    ## Half-light radius and Lya halo fitting ##
    r_min = 10. * kpc
    #r_bin = 0.5 * (r_edges[:-1] + r_edges[1:])
    #r_mask = (r_bin > r_min) & (SB_r > 0)
    #pfit = np.polyfit(r_bin[r_mask], np.log10(SB_r[r_mask]), 1)
    #R_scale, A_scale = -1. / pfit[0], pfit[1]
    #r_mask = (r_bin > r_min) & (SB_r_IGMA > 0)
    #pfit_IGMA = np.polyfit(r_bin[r_mask], np.log10(SB_r_IGMA[r_mask]), 1)
    #R_scale_IGMA, A_scale_IGMA = -1. / pfit_IGMA[0], pfit_IGMA[1]
    R_half = np.median(r_impact)
    R_half_IGMA = weighted_median(r_impact, W_IGMA)
    r_edges_LOS = np.linspace(0., SB_radius, n_r_bins+1)
    r_bin_LOS = 0.5 * (r_edges_LOS[:-1] + r_edges_LOS[1:])
    dA_LOS = 4.*np.pi * (r_edges_LOS[1:]**2 - r_edges_LOS[:-1]**2)
    SB_fac_LOS = L_α / (2.*np.pi * arcsec**2 * (1.+z)**4 * dA_LOS)
    logr_edges_LOS = np.logspace(np.log10(5e-2*kpc), np.log10(SB_radius), n_logr_bins+1)
    logr_edges_LOS[0] *= 1e-6
    logr_bin_LOS = 0.5 * (logr_edges_LOS[:-1] + logr_edges_LOS[1:])
    logdA_LOS = 4.*np.pi * (logr_edges_LOS[1:]**2 - logr_edges_LOS[:-1]**2)
    logSB_fac_LOS = L_α / (2.*np.pi * arcsec**2 * (1.+z)**4 * logdA_LOS)
    Δv_edges_LOS = np.linspace(-1e3, 1e3, n_Dv_bins+1)
    Δv_edges_LOS[0] = -1e4
    Δv_edges_LOS[-1] = 1e4
    Δv_bin_LOS = 0.5 * (Δv_edges_LOS[1:] + Δv_edges_LOS[:-1])
    dλ_LOS = (Δv_edges_LOS[1:] - Δv_edges_LOS[:-1]) * km * lambda0 * (1.+z) / c
    Flux_fac_LOS = L_α / (2.*np.pi * d_L**2 * dλ_LOS)
    #if verbose:
    #    print('n_photons =', n_photons)
    #    print('n_escaped =', n_escaped)
    #    print('L_Lya       =', L_α, 'erg/s')
    #    print('L_UV_tot    =', L_UV_tot, 'erg/s/angstrom')
    #    print('M_UV_tot    =', M_UV_tot)
    #    print('EW_Lya_INT  =', EW_α_INT, 'angstroms')
    #    print('f_esc_Lya   =', f_esc_α)
    #    print('f_esc_Lya_IGMA =', f_esc_α_IGMA)
    #    print('T_IGMA      =', T_IGMA)
    #    print('F_rb        =', F_rb)
    #    print('F_rb_IGMA   =', F_rb_IGMA)
    #    print('v_avg       =', v_avg, 'km/s')
    #    print('v_avg_IGMA  =', v_avg_IGMA, 'km/s')
    #    print('v2_avg      =', v2_avg, 'km^2/s^2')
    #    print('v2_avg_IGMA =', v2_avg_IGMA, 'km^2/s^2')
    #    print('v_std       =', v_std, 'km/s')
    #    print('v_std_IGMA  =', v_std_IGMA, 'km/s')
    #    print('len(Δv_edges) =', len(Δv_edges))
    #    print('v_peak_IGMA =', v_peak_IGMA, 'km/s')
    #    print('FWHM_IGMA   =', FWHM_IGMA, 'km/s')
    #    print('len(r_edges) =', len(r_edges))
    #    print('SB_radius   =', SB_radius/kpc, 'kpc')
    #    print('R_scale     =', R_scale/kpc, 'kpc')
    #    print('R_scale_IGMA=', R_scale_IGMA/kpc, 'kpc')
    #    print('R_half      =', R_half/kpc, 'kpc')
    #    print('R_half_IGMA =', R_half_IGMA/kpc, 'kpc')
    for i in range(n_LOS):
        print('\rProgress:', i+1, '/', n_LOS, end='')
        k = ks[i]
        #mask = (k[0]*k_esc[:,0] + k[1]*k_esc[:,1] + k[2]*k_esc[:,2] > μ_min)
        μs = k[0]*k_esc[:,0] + k[1]*k_esc[:,1] + k[2]*k_esc[:,2]
        mask = (μs > μ_min_cut)
        N_1_LOS[i] = np.count_nonzero(mask)
        red_mask_LOS = red_mask[mask]
        blue_mask_LOS = (~red_mask_LOS)
        #W_1_LOS = W_1[mask]
        W_LOS = np.exp(-(1.-μs[mask])**2/(2.*Δμ**2)) / n_photons_per_pixel
        W_IGMA_LOS = W_IGMA[mask] * W_LOS
        N_W_LOS[i] = n_photons_per_pixel * np.sum(W_LOS)
        N_W_IGMA_LOS[i] = n_photons_per_pixel * np.sum(W_IGMA_LOS)
        f_esc_α_LOS[i] = np.sum(W_LOS)
        f_esc_α_IGMA_LOS[i] = np.sum(W_IGMA_LOS)
        F_rb_LOS[i] = np.sum(W_LOS[red_mask_LOS]) / np.sum(W_LOS[blue_mask_LOS])
        F_rb_IGMA_LOS[i] = np.sum(W_IGMA_LOS[red_mask_LOS]) / np.sum(W_IGMA_LOS[blue_mask_LOS])
        freq_esc_LOS = freq_esc[mask]
        v_avg_LOS[i] = np.average(freq_esc_LOS, weights=W_LOS)
        v2_avg_LOS[i] = np.average(freq_esc_LOS**2, weights=W_LOS)
        v_std_LOS[i] = np.sqrt(v2_avg_LOS[i] - v_avg_LOS[i]**2)
        v_avg_IGMA_LOS[i] = np.average(freq_esc_LOS, weights=W_IGMA_LOS)
        v2_avg_IGMA_LOS[i] = np.average(freq_esc_LOS**2, weights=W_IGMA_LOS)
        v_std_IGMA_LOS[i] = np.sqrt(v2_avg_IGMA_LOS[i] - v_avg_IGMA_LOS[i]**2)
        ## Flux histogram ##
        hist_LOS, temp = np.histogram(freq_esc_LOS, bins=Δv_edges_LOS, weights=W_LOS)
        Flux_LOS[i] = hist_LOS * Flux_fac_LOS
        IFlux_LOS[i] = np.cumsum(Flux_LOS[i]*dλ_LOS)
        IFlux_LOS[i] /= IFlux_LOS[i,-1]
        v_peak_LOS[i], FWHM_LOS[i] = v_peak_FWHM(Δv_bin_LOS, hist_LOS)
        hist_IGMA_LOS, temp = np.histogram(freq_esc_LOS, bins=Δv_edges_LOS, weights=W_IGMA_LOS)
        Flux_IGMA_LOS[i] = hist_IGMA_LOS * Flux_fac_LOS
        IFlux_IGMA_LOS[i] = np.cumsum(Flux_IGMA_LOS[i]*dλ_LOS)
        IFlux_IGMA_LOS[i] /= IFlux_IGMA_LOS[i,-1]
        v_peak_IGMA_LOS[i], FWHM_IGMA_LOS[i] = v_peak_FWHM(Δv_bin_LOS, hist_IGMA_LOS)
        ## SB_r histogram ##
        r_impact_LOS = r_impact[mask]
        r_hist_LOS, temp = np.histogram(r_impact_LOS, bins=r_edges_LOS, weights=W_LOS)
        SB_r_LOS[i] = SB_fac_LOS * r_hist_LOS
        ISB_r_LOS[i] = np.cumsum(SB_r_LOS[i]*dA_LOS)
        ISB_r_LOS[i] /= ISB_r_LOS[i,-1]
        r_hist_IGMA_LOS, temp = np.histogram(r_impact_LOS, bins=r_edges_LOS, weights=W_IGMA_LOS)
        SB_r_IGMA_LOS[i] = SB_fac_LOS * r_hist_IGMA_LOS
        ISB_r_IGMA_LOS[i] = np.cumsum(SB_r_IGMA_LOS[i]*dA_LOS)
        ISB_r_IGMA_LOS[i] /= ISB_r_IGMA_LOS[i,-1]
        ## SB_logr histogram ##
        logr_hist_LOS, temp = np.histogram(r_impact_LOS, bins=logr_edges_LOS, weights=W_LOS)
        SB_logr_LOS[i] = logSB_fac_LOS * logr_hist_LOS
        ISB_logr_LOS[i] = np.cumsum(SB_logr_LOS[i]*logdA_LOS)
        ISB_logr_LOS[i] /= ISB_logr_LOS[i,-1]
        logr_hist_IGMA_LOS, temp = np.histogram(r_impact_LOS, bins=logr_edges_LOS, weights=W_IGMA_LOS)
        SB_logr_IGMA_LOS[i] = logSB_fac_LOS * logr_hist_IGMA_LOS
        ISB_logr_IGMA_LOS[i] = np.cumsum(SB_logr_IGMA_LOS[i]*logdA_LOS)
        ISB_logr_IGMA_LOS[i] /= ISB_logr_IGMA_LOS[i,-1]
        ## Half-light radius and Lya halo fitting ##
        r_mask_LOS = (r_bin_LOS > r_min) & (SB_r_LOS[i] > 0)
        pfit_LOS = np.polyfit(r_bin_LOS[r_mask_LOS], np.log10(SB_r_LOS[i,r_mask_LOS]), 1)
        R_scale_LOS[i], A_scale_LOS[i] = -1. / pfit_LOS[0], pfit_LOS[1]
        r_mask_LOS = (r_bin_LOS > r_min) & (SB_r_IGMA_LOS[i] > 0)
        pfit_IGMA_LOS = np.polyfit(r_bin_LOS[r_mask_LOS], np.log10(SB_r_IGMA_LOS[i,r_mask_LOS]), 1)
        R_scale_IGMA_LOS[i], A_scale_IGMA_LOS[i] = -1. / pfit_IGMA_LOS[0], pfit_IGMA_LOS[1]
        R_half_LOS[i] = weighted_median(r_impact_LOS, W_LOS)
        R_half_IGMA_LOS[i] = weighted_median(r_impact_LOS, W_IGMA_LOS)
        ## F_red / F_blue as a function of radius ##
        F_red_r_LOS[i], temp = np.histogram(r_impact_LOS[red_mask_LOS], bins=r_edges_LOS, weights=W_LOS[red_mask_LOS])
        F_blue_r_LOS[i], temp = np.histogram(r_impact_LOS[blue_mask_LOS], bins=r_edges_LOS, weights=W_LOS[blue_mask_LOS])
        F_red_r_IGMA_LOS[i], temp = np.histogram(r_impact_LOS[red_mask_LOS], bins=r_edges_LOS, weights=W_IGMA_LOS[red_mask_LOS])
        F_blue_r_IGMA_LOS[i], temp = np.histogram(r_impact_LOS[blue_mask_LOS], bins=r_edges_LOS, weights=W_IGMA_LOS[blue_mask_LOS])
    print()
    L_α_LOS = L_α * f_esc_α_LOS
    L_α_IGMA_LOS = L_α * f_esc_α_IGMA_LOS
    EW_α_LOS = L_α_LOS / L_UV_LOS
    EW_α_IGMA_LOS = L_α_IGMA_LOS / L_UV_LOS
    f_esc_UV = np.median(f_esc_UV_LOS)
    EW_α = np.median(EW_α_LOS)
    EW_α_IGMA = np.median(EW_α_IGMA_LOS)
    if verbose:
        print('f_esc_UV    =', f_esc_UV)
        print('EW_Lya      =', EW_α, 'angstroms')
        print('EW_Lya_IGMA =', EW_α_IGMA, 'angstroms')
    with h5py.File(fesc_filename,'w') as f:
        f.attrs['z'] = z
        f.attrs['d_L'] = d_L
        f.attrs['SB_radius'] = SB_radius
        f.attrs['L_Lya'] = L_α
        f.attrs['L_UV_tot'] = L_UV_tot
        f.attrs['f_esc_UV'] = f_esc_UV
        f.attrs['f_esc_Lya'] = f_esc_α
        f.attrs['f_esc_Lya_IGMA'] = f_esc_α_IGMA
        f.attrs['T_IGMA'] = T_IGMA
        f.attrs['EW_Lya_INT'] = EW_α_INT
        f.attrs['EW_Lya'] = EW_α
        f.attrs['EW_Lya_IGMA'] = EW_α_IGMA
        f.attrs['F_rb'] = F_rb
        f.attrs['F_rb_IGMA'] = F_rb_IGMA
        f.attrs['v_avg'] = v_avg
        f.attrs['v_avg_IGMA'] = v_avg_IGMA
        f.attrs['v2_avg'] = v2_avg
        f.attrs['v2_avg_IGMA'] = v2_avg_IGMA
        f.attrs['v_std'] = v_std
        f.attrs['v_std_IGMA'] = v_std_IGMA
        f.attrs['v_com_mag'] = v_com_mag
        #f.attrs['v_peak'] = v_peak
        #f.attrs['v_peak_IGMA'] = v_peak_IGMA
        #f.attrs['FWHM'] = FWHM
        #f.attrs['FWHM_IGMA'] = FWHM_IGMA
        #f.attrs['R_scale'] = R_scale
        #f.attrs['A_scale'] = A_scale
        #f.attrs['R_scale_IGMA'] = R_scale_IGMA
        #f.attrs['A_scale_IGMA'] = A_scale_IGMA
        f.attrs['R_half'] = R_half
        f.attrs['R_half_IGMA'] = R_half_IGMA
        f.create_dataset('v_com', data=v_com)
        f.create_dataset('L_UV_LOS', data=L_UV_LOS)
        f.create_dataset('M_UV_LOS', data=M_UV_LOS)
        f.create_dataset('f_esc_UV_LOS', data=f_esc_UV_LOS)
        f.create_dataset('N_1_LOS', data=N_1_LOS)
        f.create_dataset('N_W_LOS', data=N_W_LOS)
        f.create_dataset('N_W_IGMA_LOS', data=N_W_IGMA_LOS)
        f.create_dataset('f_esc_Lya_LOS', data=f_esc_α_LOS)
        f.create_dataset('f_esc_Lya_IGMA_LOS', data=f_esc_α_IGMA_LOS)
        f.create_dataset('T_IGMA_LOS', data=f_esc_α_IGMA_LOS/f_esc_α_LOS)
        f.create_dataset('EW_Lya_LOS',    data=EW_α_LOS)
        f.create_dataset('EW_Lya_IGMA_LOS',    data=EW_α_IGMA_LOS)
        f.create_dataset('F_rb_LOS',      data=F_rb_LOS)
        f.create_dataset('F_rb_IGMA_LOS', data=F_rb_IGMA_LOS)
        f.create_dataset('v_avg_LOS',      data=v_avg_LOS)
        f.create_dataset('v_avg_IGMA_LOS', data=v_avg_IGMA_LOS)
        f.create_dataset('v2_avg_LOS',      data=v2_avg_LOS)
        f.create_dataset('v2_avg_IGMA_LOS', data=v2_avg_IGMA_LOS)
        f.create_dataset('v_std_LOS',      data=v_std_LOS)
        f.create_dataset('v_std_IGMA_LOS', data=v_std_IGMA_LOS)
        f.create_dataset('v_peak_LOS', data=v_peak_LOS)
        f.create_dataset('v_peak_IGMA_LOS', data=v_peak_IGMA_LOS)
        f.create_dataset('FWHM_LOS', data=FWHM_LOS)
        f.create_dataset('FWHM_IGMA_LOS', data=FWHM_IGMA_LOS)
        f.create_dataset('R_scale_LOS', data=R_scale_LOS)
        f.create_dataset('A_scale_LOS', data=A_scale_LOS)
        f.create_dataset('R_scale_IGMA_LOS', data=R_scale_IGMA_LOS)
        f.create_dataset('A_scale_IGMA_LOS', data=A_scale_IGMA_LOS)
        f.create_dataset('R_half_LOS', data=R_half_LOS)
        f.create_dataset('R_half_IGMA_LOS', data=R_half_IGMA_LOS)
        ## Averaged arrays ##
        #f.create_dataset('Dv_edges', data=Δv_edges)
        #f.create_dataset('Flux', data=Flux)
        #f.create_dataset('IFlux', data=IFlux)
        #f.create_dataset('Flux_IGMA', data=Flux_IGMA)
        #f.create_dataset('IFlux_IGMA', data=IFlux_IGMA)
        #f.create_dataset('r_edges', data=r_edges)
        #f.create_dataset('SB_r', data=SB_r)
        #f.create_dataset('ISB_r', data=ISB_r)
        #f.create_dataset('SB_r_IGMA', data=SB_r_IGMA)
        #f.create_dataset('ISB_r_IGMA', data=ISB_r_IGMA)
        #f.create_dataset('F_red_r', data=F_red_r)
        #f.create_dataset('F_red_r_IGMA', data=F_red_r_IGMA)
        #f.create_dataset('F_blue_r', data=F_blue_r)
        #f.create_dataset('F_blue_r_IGMA', data=F_blue_r_IGMA)
        ## LOS arrays ##
        f.create_dataset('r_edges_LOS', data=r_edges_LOS)
        f.create_dataset('logr_edges_LOS', data=logr_edges_LOS)
        f.create_dataset('Dv_edges_LOS', data=Δv_edges_LOS)
        f.create_dataset('Flux_LOS', data=Flux_LOS)
        f.create_dataset('IFlux_LOS', data=IFlux_LOS)
        f.create_dataset('Flux_IGMA_LOS', data=Flux_IGMA_LOS)
        f.create_dataset('IFlux_IGMA_LOS', data=IFlux_IGMA_LOS)
        f.create_dataset('SB_r_LOS', data=SB_r_LOS)
        f.create_dataset('ISB_r_LOS', data=ISB_r_LOS)
        f.create_dataset('SB_r_IGMA_LOS', data=SB_r_IGMA_LOS)
        f.create_dataset('ISB_r_IGMA_LOS', data=ISB_r_IGMA_LOS)
        f.create_dataset('SB_logr_LOS', data=SB_logr_LOS)
        f.create_dataset('ISB_logr_LOS', data=ISB_logr_LOS)
        f.create_dataset('SB_logr_IGMA_LOS', data=SB_logr_IGMA_LOS)
        f.create_dataset('ISB_logr_IGMA_LOS', data=ISB_logr_IGMA_LOS)
        f.create_dataset('F_red_r_LOS', data=F_red_r_LOS)
        f.create_dataset('F_red_r_IGMA_LOS', data=F_red_r_IGMA_LOS)
        f.create_dataset('F_blue_r_LOS', data=F_blue_r_LOS)
        f.create_dataset('F_blue_r_IGMA_LOS', data=F_blue_r_IGMA_LOS)

def write_snaps(snap_beg, snap_end, v_snap_beg=42, n_exp=1, emission='rec', verbose=False):
    snaps = range(snap_beg,snap_end+1)
    n_snaps = len(snaps)
    i_snaps = range(snap_beg-v_snap_beg, snap_end-v_snap_beg+1)
    v_com_filename = 'HiZFIRE_v_com.h5'
    with h5py.File(v_com_filename,'r') as f:
        if f.attrs['N_vir'] < 3: raise ValueError('N_vir should be > 3')
        z = f['z'][i_snaps]
        M_vir = f['M_vir'][i_snaps]
        M_gas = f['M_gas'][i_snaps]
        M_star = f['M_star'][i_snaps]
        M_dm = f['M_dm'][i_snaps]
        R_vir = f['R_vir'][i_snaps]
        v_com_gas = f['v_com_gas'][i_snaps]
        v_com_star = f['v_com_star'][i_snaps]
        v_com_dm = f['v_com_dm'][i_snaps]
    if verbose:
        print('z =', z)
        print('M_vir =', M_vir, 'Msun')
        print('M_gas =', M_gas, 'Msun = ', M_gas/M_vir, 'M_vir')
        print('M_star =', M_star, 'Msun = ', M_star/M_vir, 'M_vir')
        print('M_dm =', M_dm, 'Msun = ', M_dm/M_vir, 'M_vir')
        print('R_vir =', R_vir, 'kpc')
        print('|v_com_gas| =', np.sqrt(np.sum(v_com_gas**2,axis=1)), 'km/s')
        print('|v_com_star| =', np.sqrt(np.sum(v_com_star**2,axis=1)), 'km/s')
        print('|v_com_dm| =', np.sqrt(np.sum(v_com_dm**2,axis=1)), 'km/s')
        print('|v_com_dm - v_com_gas| =', np.sqrt(np.sum((v_com_dm-v_com_gas)**2,axis=1)), 'km/s')
        print('|v_com_dm - v_com_star| =', np.sqrt(np.sum((v_com_dm-v_com_star)**2,axis=1)), 'km/s')

    for i in range(n_snaps):
        write_snap(snaps[i], n_exp=n_exp, v_com=v_com_dm[i], emission=emission, verbose=(True if (i==0) else verbose))

em = 'rec'
if len(sys.argv) == 1:
    snap_beg,snap_end = 42,67
elif len(sys.argv) == 2:
    snap_beg,snap_end = int(sys.argv[1]),int(sys.argv[1])
elif len(sys.argv) == 3:
    if sys.argv[2] in ['rec','col','tot']:
        snap_beg,snap_end,em = int(sys.argv[1]),int(sys.argv[1]),sys.argv[2]
    else:
        snap_beg,snap_end = int(sys.argv[1]),int(sys.argv[2])
elif len(sys.argv) == 4:
    snap_beg,snap_end,em = int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
else:
    raise ValueError('Expecting 0, 1, 2, or 3 arguments.')

write_snaps(snap_beg, snap_end, n_exp=4, emission=em, verbose=False)
#for snap in [45, 57, 67]:
#    write_snaps(snap, snap, n_exp=360, emission='col', verbose=True)

