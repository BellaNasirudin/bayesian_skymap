#!/usr/bin/env python
# coding: utf-8
import numpy as np
# import pygdsm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import healpy as hp
from astropy import units as un
from astropy import constants as const
from scipy.special import sph_harm
from scipy.sparse import linalg, dia_array, diags
from scipy.stats import Covariance
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares
import sys
import multiprocessing as mp
import time
import warnings
import emcee
from multiprocessing import Pool, Process
from concurrent.futures import ThreadPoolExecutor

def estimate_diag_precond(lhs_op, size):
    diag = np.zeros(size)
    for i in range(size):
        e = np.zeros(size)
        e[i] = 1.0
        diag[i] = lhs_op(e)[i]
    return diag

def get_lmax_mmax_lenalms(nside, lmax=None):
    if lmax == None:
        lmax = 3 * nside -1
    else:
        lmax = lmax
    mmax = lmax
    
    len_alms = ((lmax+1)*(lmax+2) - (lmax-mmax)*(lmax-mmax+1))//2

    return lmax, mmax, len_alms

def m_tilde(g, op_mtrx, s):
    K = s[:, None] * op_mtrx

    return K @ g # +1

def m_tilde_t(v, op_mtrx, s):
    K = s[:, None] * op_mtrx

    return np.transpose(K) @ v

def g_rhs_vec(s0, data, covN, covG, operator, op_alm, mu = 0, nside=16, lmax = 10):

    invN = 1 / covN

    #draw random vectors omega_n and omega_a
    omega_n = np.random.normal(0, 1, np.shape(data))
    omega_g = np.random.normal(0, 1, len(covG)) # np.shape(data)) #

    data_ = data.copy()


    if op_alm == True:
        len_alms = 2 * get_lmax_mmax_lenalms(nside, lmax = lmax)[-1]
        first_term = np.zeros(len_alms)
        second_term = np.zeros(len_alms)
        x = hp.map2alm(1 / np.transpose(s0) * invN * data - 1, lmax=lmax)      
        y = hp.map2alm(1 / np.transpose(s0) * (invN**0.5) * omega_n - 1, lmax=lmax)  

        first_term[:int(len_alms/2)] = np.real(x)
        first_term[int(len_alms/2):] = np.imag(x)

        second_term[:int(len_alms/2)] = np.real(y)
        second_term[int(len_alms/2):] = np.imag(y) 
    else:
        
        first_term = m_tilde_t(invN * data_, operator, s0)
        second_term = m_tilde_t(invN**0.5 * omega_n, operator, s0)

    if covG.ndim == 2:
        invG = np.linalg.inv(covG)
        third_term = invG**0.5 @ omega_g #(m_tilde_t(omega_g, operator, 1) + 1)
    else:
        invG = 1 / (covG)
        invG[np.isinf(invG)] = 0
        third_term = invG**0.5 * omega_g

    if op_alm == True:
        mu = np.random.normal(mu, covG, len(mu))
        return first_term + second_term + third_term + invG * mu 
    else:
        # mu = np.random.normal(mu, np.diag(covG), len(mu))
        # print(first_term, second_term, third_term)
        return first_term + second_term + third_term #+ invG @ mu 

def g_lhs_matrix(g, s0, covN, covG, operator):
    s0_= s0.copy()

    invN = 1 / covN
    
    first_term =  m_tilde_t(invN * m_tilde(g, operator, s0_), operator, s0_)
        
    second_term = np.linalg.inv(covG) @ g
    
    return first_term + second_term

def g_lhs_healpy(gain, s0, covN, covG, nside, len_alms, lmax=None):
 
    invN = 1 / covN
    first_term =  np.zeros(len_alms)
    x = hp.map2alm((1 / np.transpose(s0) * invN * s0 * (1 + hp.alm2map(gain[:int(len_alms/2)] + 1j * gain[int(len_alms/2):], nside=nside))) - 1, lmax=lmax) 
    first_term[:int(len_alms/2)] = np.real(x)
    first_term[int(len_alms/2):] = np.imag(x)

    invG = 1 / (covG)
    invG[np.isinf(invG)] = 0      
    second_term = invG * gain

    return first_term + second_term


# GCR where the data $d$ is given by $\mathbf{d}= \tilde{\mathbf{m}}_j\mathbf{s}_{true} +\mathbf{n}$ where $\tilde{\mathbf{m}}_j = (1+\mathbf{Y} \cdot \mathbf{a}_j) \mathbf{B}_j$ (when $j=H$, $(a_j=a, b_j=b_H$) and when $j=x$, $(a_j=0, b_j=b_x$)
# 
# $(\tilde{\mathbf{m}}^T \mathbf{N}^{-1}\tilde{\mathbf{m}} + \mathbf{S}^{-1}) \mathbf{s} = \tilde{\mathbf{m}}^T\mathbf{N}^{-1}\mathbf{d} + \tilde{\mathbf{m}}^T \mathbf{N}^{-1/2}\omega_n + \mathbf{S}^{-1/2}\omega_s$

def beam_window(fwhm, lmax, ncells=200, extent=5):

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sky_coords = np.linspace(0, fwhm*extent, ncells)

    attenuation = np.exp(
        - 0.5 * (sky_coords ** 2) * 1. / ( sigma ** 2)) #* 1 / (sigma * np.sqrt(2* np.pi))

    bl = hp.sphtfunc.beam2bl(attenuation, sky_coords, lmax=lmax)
    bl *=  1 / np.max(bl)

    return bl

def get_beam_vec(nside, beam_fwhm, lmax=None):
    lmax, mmax, len_alms = get_lmax_mmax_lenalms(nside, lmax=lmax)
    
    amps = [1.0, 1j]
    alms_ = np.zeros(len_alms, dtype=np.complex128)
    sigma = beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ell = np.arange(lmax + 1.0)
    factor = np.exp(-0.5 * (ell * (ell + 1)) * sigma ** 2)
    
    beam_vec = np.zeros(int(len_alms), dtype=np.complex128)
    
    for ii in range(len_alms):
        alms_ *= 0
        for jj in range(len(amps)):
            alms_[ii] = amps[jj]
            x = hp.almxfl(alms_, factor, inplace = True)
            beam_vec[ii] += x[ii]

    return beam_vec

def beam_window_transpose(beam_vec):
    return (np.real(beam_vec)*np.sum(np.real(beam_vec)) + np.imag(beam_vec)*np.sum(np.imag(beam_vec))) / np.sum(beam_vec)

def transpose_downsampling(map_low, nside_high, power=0):
    """
    Transpose of ud_grade from nside_high → nside_low (downsampling).
    This maps from low-res map to high-res (spread values to children).
    """
    nside_low = hp.get_nside(map_low)
    factor = nside_high // nside_low
    assert nside_high % nside_low == 0, "NSIDE must divide evenly"

    npix_high = hp.nside2npix(nside_high)
    map_high = np.zeros(npix_high)

    for i_low in range(len(map_low)):
        # Get children pixel indices
        base_nest = hp.ring2nest(nside_low, i_low) * (factor**2)
        child_nests = base_nest + np.arange(factor**2)
        child_rings = hp.nest2ring(nside_high, child_nests)

        if power == 0:
            weight = 1.0 / (factor**2)
        elif power == -2:
            weight = (factor)**2 / (factor**2)  # cancels → 1
        else:
            raise ValueError("Unsupported power value")

        map_high[child_rings] = map_low[i_low] * weight

    return map_high
    
def transpose_beam(vec, freqs, ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask=None):
    const = (freqs / ref_freq)**beta
    
    ret =  hp.smoothing(vec , lmax=lmax, fwhm=(beam_fwhm * freq_beam/ freqs)) #beam_window=beam_vec_t)
    
    ret[pixel_mask == 0] = 0

    return transpose_downsampling(ret, nside) * const

def apply_beam(amps, freqs, ref_freq,  beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask=None):
    amps_ = amps.copy()
    
    amps_ = hp.pixelfunc.ud_grade(amps_ * (freqs / ref_freq) ** beta, nside_new) 
    amps_[pixel_mask==0] = hp.UNSEEN
    ret = hp.smoothing(amps_, fwhm=(beam_fwhm * freq_beam/ freqs), lmax=lmax)
    
    return ret

def amps_lhs(amps, gain, all_covN, covA, freqs, ref_freq, beta, beam_vec_t, nside, nside_new, Nfreqs, beam_fwhm, lmax, freq_beam, 
    pixel_mask=None, parallel=True):
    '''
    LHS matrix multiplication with vector amps
    
    '''
    npix = hp.nside2npix(nside)

    amps_npix = amps.copy()

    invN1 = 1 / (all_covN[0])
    invN2 = 1 / (all_covN[1])
    
    invN1[pixel_mask==0] = 0

    if parallel == True:
        def process_ff(ff):
            if ff < Nfreqs - 1:
                return transpose_beam(
                    invN1 * apply_beam(
                        amps_npix, freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask
                    ),
                    freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask
                )
            else:
                scale = gain * ((freqs[ff] / ref_freq) ** beta)
                return scale * invN2 * (gain * amps_npix * (freqs[ff] / ref_freq) ** beta)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_ff, range(Nfreqs)))

        v = np.sum(results, axis=0)

    elif parallel == False:

        v = np.zeros(npix)

        for ff in range(Nfreqs):
            if ff<(Nfreqs-1):
                v += transpose_beam(invN1 * apply_beam(amps_npix, freqs[ff], ref_freq,  beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask=pixel_mask)
                    , freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask)
            else:
                v += gain * ((freqs[ff] / ref_freq) ** beta) * invN2 * (gain * amps_npix * (freqs[ff] / ref_freq) ** beta)

    second_term = 1 / covA
    
    second_term[np.isnan(second_term)] = 0
    second_term[np.isinf(second_term)] = 0

    return v +  second_term * amps_npix

def amps_rhs(all_data, gain, all_covN, covA, freqs, ref_freq, beta, beam_vec_t, nside, nside_new, Nfreqs, beam_fwhm, lmax, freq_beam, pixel_mask=None, sprior_mean=None):
    '''
    RHS vector
    '''
    invN1 = 1 / (all_covN[0])
    invN2 = 1 / (all_covN[1])
    
    invN1[pixel_mask==0] = 0

    #draw random vectors omega_n and omega_a
    omega_a = np.random.normal(0, 1, np.shape(all_data[1])[0])

    first_term = np.zeros(np.shape(all_data[1])[0])
    second_term = np.zeros(np.shape(all_data[1])[0])
    third_term = np.zeros(np.shape(all_data[1])[0])   
    
    for ff in range(Nfreqs):
        if ff<(Nfreqs-1):
            omega_n = np.random.normal(0, 1, np.shape(all_data[0])[1])

            omega_n[pixel_mask==0] = 0

            first_term += transpose_beam( invN1 * all_data[0][ff], freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask)
            second_term += transpose_beam( invN1**0.5 * omega_n, freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax, nside, nside_new, pixel_mask)
        else:
            omega_n = np.random.normal(0, 1, np.shape(all_data[1])[0])
            first_term += gain * ((freqs[ff] / ref_freq) ** beta) * (invN2 * all_data[1])
            second_term += gain * ((freqs[ff] / ref_freq) ** beta) * (invN2**0.5 * omega_n[ff]) #
 
    third_term = 1 / (covA)**0.5 * omega_a
    third_term[np.isnan(third_term)] = 0
    third_term[np.isinf(third_term)] = 0

    return first_term + second_term + third_term + 1 / (covA) * sprior_mean

def calc_model_spectral(amps, freqs, gain, beta, ndata1, beam_fwhm, ref_freq, freq_beam, lmax_beam=None, nside=128, nside_new=16, pixel_mask=None):

    amps_npix = amps.copy()

    def compute_model1(ff):
        return apply_beam(amps_npix, freqs[ff], ref_freq, beta, beam_fwhm, freq_beam, lmax_beam, nside, nside_new, pixel_mask=pixel_mask
        )

    with ThreadPoolExecutor() as executor:
        model1 = list(executor.map(compute_model1, range(ndata1)))

    model1 = np.array(model1)
    model2 = gain * (amps_npix * (freqs[-1] / ref_freq) ** beta)
    return [model1, model2]

def precompute_conv(amps, freqs, gain, ndata1, beam_fwhm, freq_beam, lmax_beam, op_spect, nside=128, nside_new=16):

    ndim = np.shape(op_spect)[-1]

    model1 = np.zeros((ndim, ndata1, hp.nside2npix(nside_new)))
    
    for ss in range(ndim):
        for ff in range(ndata1):            
            amps_ = amps.copy()
            mask = op_spect[:, ss] == 0
            amps_[mask] = 0 
            x = hp.smoothing(hp.pixelfunc.ud_grade(amps_, nside_new), fwhm=(beam_fwhm * freq_beam / freqs[ff]), lmax=lmax_beam)
            
            x[x== hp.UNSEEN] = 0
            model1[ss, ff] += x
            
    model2 = gain * (amps)
    
    return [model1, model2] #np.concatenate((np.sum(model1, axis=0), model2[np.newaxis, :]), axis=0)

def log_prob_haslam(beta, data, precomp, freqs, ref_freq, op_spect, var,
                    amps, gain, beam_fwhm, lmax_beam, freq_beam,
                    nside, nside_new, sigfig=None, pixel_mask=None, prior=False):

    nfreq = len(freqs)
    npix = np.shape(data[1])[0]
    npix_new = hp.nside2npix(nside_new)

    # Apply operator to beta if needed
    beta_ = op_spect @ beta if op_spect.ndim == 2 else beta.copy()

    # Compute model
    model = calc_model_spectral(amps, freqs, gain, beta_, nfreq - 1, beam_fwhm, ref_freq, freq_beam=freq_beam, lmax_beam=lmax_beam, 
        nside=nside, nside_new=nside_new, pixel_mask=pixel_mask
    )

    # model1 = np.zeros((nfreq, np.shape(data[0])[1]))  

    # for ff in range(nfreqs):
    #     if ff<len(freqs)-1:
    #         for nn in range(len(precomp[0])):
    #             if len(np.shape(op_spect)) == 2:
    #                 mask = op_spect[:, nn].astype(bool)
    #                 beta_val = np.mean(beta_[mask])
    #             else:
    #                 beta_val = beta_

    #             model[ff:npix_new] += precomp[0][nn, ff] * (freqs[ff] / ref_freq)** (beta_val)
    #     else:
    #         model2 = precomp[1] * (freqs[ff] / ref_freq)** (beta_)
    # model = [model1, model2]

    x = np.zeros((nfreq, npix))
    y = np.zeros((nfreq, npix)) if prior else None

    def compute_residual(ff):
        res_x = np.zeros(npix)
        res_y = np.zeros(npix) if prior else None
        if ff < (nfreq - 1):
            res_x[:npix_new] = (data[0][ff] - model[0][ff])**2 / var[0]
            if prior:
                res_y[:] = 1 / var[0] * (freqs[ff]/ref_freq)**beta_ * np.log(freqs[ff]/ref_freq)
        else:
            res_x[:] = (data[1] - model[1])**2 / var[1]
            if prior:
                res_y[:] = 1 / var[1] * (freqs[ff]/ref_freq)**beta_ * np.log(freqs[ff]/ref_freq)
        return ff, res_x, res_y

    with ThreadPoolExecutor() as executor:
        for ff, res_x, res_y in executor.map(compute_residual, range(nfreq)):
            x[ff] = res_x
            if prior:
                y[ff] = res_y

    prob = -0.5 * np.sum(x)
    if prior:
        prob += np.log(np.sqrt(np.sum(y**2)))

    if sigfig is not None:
        prob = np.round(prob, sigfig) * 10**sigfig

    return prob
