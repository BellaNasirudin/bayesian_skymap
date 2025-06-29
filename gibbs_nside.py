#!/usr/bin/env python
# coding: utf-8
import numpy as np
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
import sys
import multiprocessing as mp
import time
import warnings
from bayesian_func import *

fdir = "/Users/bellanasirudin/Data/" # "/nvme1/scratch/anasir/data/" # 
fout = fdir

op_alm = bool(int(sys.argv[1]))

ntimes = int(sys.argv[2])
nstart = int(sys.argv[3])

file_s0 = str(sys.argv[4])
beam_deg = int(sys.argv[5])

sig1 = int(sys.argv[6])
sig2 = int(sys.argv[7])

try:
    sig3 = int(sys.argv[8])
    print("CONFUSED NOISE LEVEL %i" %sig3, flush=True)
except:
    sig3 = sig2

print(op_alm, flush=True)
print(ntimes, nstart, file_s0, beam_deg, flush=True)

if nstart == 0:
    file_s0 = None

nside = 128
if beam_deg == 10:
    nside_new = 16
elif beam_deg == 30:
    nside_new = 8

beam_fwhm = np.deg2rad(beam_deg)
data = np.load(fdir + "data_haslam_new_nside%i_fwhm%ideg_nside%i.npz" %(nside, beam_deg, nside_new))

data1 = data["sky_lowres"]

freqs = data["freq"]

beta = data["beta"]
lmax_beam = data["lmax_beam"]
print("LMAX BEAM IS ", lmax_beam)
print("USING %i FREQUENCY BINS" %len(freqs))

true_s = data["sky_haslam"]

npix = hp.nside2npix(nside)

lmax_gain = 5

ref_freq = data["ref_freq"]

indx_ref = np.where(freqs== ref_freq)[0][0]
freq_beam = data["freq_beam"]

if op_alm == False:
    data2 = data["sky_gain"]
    gain = data["gain"]
    operator = data["operator"]
    covG = np.eye(np.shape(operator)[-1]) * (0.15 * (1 + gain))**2 #  0.05 **2 #  0.05**2 #
else:
    len_alms = get_lmax_mmax_lenalms(nside, lmax = lmax_gain)[-1]

    try:
        gain = np.load(fdir + "gain_alm_lmax%i_nside%i.npy" %(lmax_gain, nside))
    except:
        print("Generating gain")
        gain = np.random.normal(0, 0.05, (2, len_alms))
        gain[1, :int(lmax_gain+1)] = 0
        np.save(fdir + "gain_alm_lmax%i_nside%i" %(lmax_gain, nside), gain)
    
    data2 = data["sky_haslam"] * ( 1 + hp.alm2map(gain[0] + gain[1] * 1j , nside=nside))
    
    covG = np.ones(2*len_alms) * (0.15 * gain.flatten())**2
    operator = None

print(1 + gain, flush=True)

# generate noise for data
sig_noise1 = sig1 * 1e-3
sig_noise2 = sig2 * 1e-3
sig_noise3 = sig3 * 1e-3


noise1 = sig_noise1**2
noise2 = sig_noise2**2
noise3 = sig_noise3**2

try:
    noise_data = np.load(fdir + "actual_noise_data_sig1%i_sig2%imK_nside%i_nside%i.npz" %(sig1, sig3, nside, nside_new))
    x1 = noise_data["x1"]
    x2 = noise_data["x2"]
    data1+= x1
    data2+= x2
except:
    print("Generating Noise")
    x1 = np.random.normal(0, sig_noise1, np.shape(data1))
    x2 = np.random.normal(0, sig_noise2, np.shape(data2))
    np.savez(fdir + "actual_noise_data_sig1%i_sig2%imK_nside%i_nside%i" %(sig1, sig3, nside, nside_new), sig1 =sig1, sig2=sig3, sig_noise1=sig_noise1, sig_noise2=sig_noise3, x1=x1, x2=x2)

    data1+= x1
    data2+= x2

# these need to be in matrix form but they're actually diagonal matrix so keep it as vector for now
covN1 = np.ones(hp.nside2npix(nside_new))
covN2 = np.ones(npix)

covN1 *= noise1
covN2 *= noise2

print("min and max amp noise ", np.min(x1), np.max(x1), np.min(x2), np.max(x2))

def gibbs_sampling(data1, data2, covN1, covN2, covG, ntimes, operator, freqs, beta, true_s, ref_freq, indx_ref, beam_fwhm, nstart=0, file_s0 = None, mu_g = 0, nside=128, nside_new=16, lmax_gain=10, lmax_beam = 150, op_alm=False, freq_beam=100):

    all_data = [data1, data2] #np.concatenate((data1, data2[np.newaxis, :]), axis=0)
    
    all_covN = [covN1, covN2]

    # set initial s
    true_s = true_s.copy()

    if nstart == 0:
        s0 = true_s.copy() #+ np.random.normal(0, 10, true_s.shape)
    else:
        try:
            s0 = np.load(fout + file_s0)["all_s"][-1]
        except:
            print("CANNOT FIND FILE FOR S0 TO CONTINUE SAMPLING FROM NSTART=" %nstart)
            raise SystemExit

    covA = (0.1 * true_s)**2
    covS = covA
    
    all_model_global = np.zeros((ntimes, np.shape(all_data[0])[0], np.shape(all_data[0])[1]))
    all_model_haslam = np.zeros((ntimes, np.shape(all_data[1])[0]))

    all_s = np.zeros((ntimes, np.shape(all_data[1])[0]))
    
    if op_alm == False:
        all_g = np.zeros((ntimes, np.shape(operator)[-1]))
    else:
        len_alms = 2 * get_lmax_mmax_lenalms(nside, lmax = lmax_gain)[-1]
        all_g = np.zeros((ntimes, len_alms ))
        
    chisq = np.zeros((ntimes, len(freqs), npix))
    ln_post = np.zeros(ntimes)

    beam_vec = np.array([beam_window(beam_fwhm*frac, lmax_beam, extent=5) for frac in (freqs[:-1]/freq_beam)])
    
    for ii in range(ntimes):
        print("Iteration ", ii , flush=True)

        ########## First part of Gibbs sampling: p(a|...) ##############

        time_start_it = time.perf_counter()

        # first just do the one for g, but only input s at higher frequencies corresponding to Haslam map
        rhs_eqn_g = g_rhs_vec(s0, data2, covN2, covG, operator, op_alm, mu = mu_g, nside=nside, lmax = lmax_gain)

        def lhs_op_gain(g):
            if op_alm == False:
                return g_lhs_matrix(g, s0, covN2, covG, operator)
            else:
                return g_lhs_healpy(g, s0, covN2, covG, nside, len_alms, lmax = lmax_gain)
            
        if op_alm == False:
            lhs_matrix_g = linalg.LinearOperator(matvec=lhs_op_gain, shape=(np.shape(operator)[-1], np.shape(operator)[-1]))  
        else:
            lhs_matrix_g = linalg.LinearOperator(matvec=lhs_op_gain, shape=(len_alms, len_alms))            
        
        if op_alm == False:
            lhs_val = g_lhs_matrix(mu_g, s0, covN2, covG, operator)
        else:
            lhs_val = g_lhs_healpy(mu_g, s0, covN2, covG, nside, len_alms, lmax = lmax_gain)

        print("lhs ", np.min(lhs_val), np.max(lhs_val), flush=True)
        print("rhs ", np.min(rhs_eqn_g), np.max(rhs_eqn_g), flush=True)
        print("min and max of lhs - rhs", np.min(lhs_val - rhs_eqn_g), np.max(lhs_val - rhs_eqn_g), flush=True)

        # time_start_it = time.perf_counter()
        # g, info = linalg.cgs(lhs_matrix_g, rhs_eqn_g, maxiter=int(1e3), rtol=1e-9)#, atol=1e-1)#, x0 = np.random.normal(0, 0.05, len(mu_g))) # 
        # print(g, info)
        # time_end = time.perf_counter()
        # print("gain: cgs took ", time_end - time_start_it, flush=True)

        # time_start_it = time.perf_counter()
        # g, info = linalg.bicgstab(lhs_matrix_g, rhs_eqn_g, maxiter=int(2e3), rtol=1e-9, atol=1e-1)#, x0 = np.random.normal(0, 0.05, len(mu_g))) # 
        # print(g, info)
        # time_end = time.perf_counter()
        # print("gain: bicgstab took ", time_end - time_start_it, flush=True)

        time_start_it = time.perf_counter()
        g, info = linalg.gmres(lhs_matrix_g, rhs_eqn_g, maxiter=int(1e3))#, rtol=1e-9, atol=1e-1)#, x0 = np.random.normal(0, 0.05, len(mu_g))) # 
        print(g, info)
        time_end = time.perf_counter()
        print("gain: gmres took ", time_end - time_start_it, flush=True)

        # time_start_it = time.perf_counter()
        # g, info = linalg.minres(lhs_matrix_g, rhs_eqn_g, maxiter=int(2.5e3), rtol=1e-9)#, x0 = np.random.normal(0, 0.05, len(mu_g))) # 
        # print(g, info)
        # time_end = time.perf_counter()
        # print("gain: minres took ", time_end - time_start_it, flush=True)

        # raise SystemExit
        if info>0:
            warnings.warn("Warning! Gain info: %i" %info)
            print("drawing gain again", flush=True)
            rhs_eqn_g = g_rhs_vec(s0, data2, covN2, covG, operator, op_alm, mu = mu_g, nside=nside, lmax = lmax_gain)
            g, info = linalg.cgs(lhs_matrix_g, rhs_eqn_g, maxiter=int(1e3), rtol=1e-9)#, x0 = np.random.normal(0, 0.05, np.shape(mu_g)))
            time_end = time.perf_counter()
            print(g, info)
            print("gain: cgs took ", time_end - time_start_it, flush=True)

        all_g[ii] = g

        if op_alm == True:
            gain = 1 + hp.alm2map(g[:int(len_alms/2)] + 1j * g[int(len_alms/2):], nside, lmax=lmax_gain)
        else:
            gain = operator @ g # + 1

        ########## second part of Gibbs sampling: p(amps|...) ##############
        time_start = time.perf_counter()

        rhs_eqn = amps_rhs(all_data, gain, all_covN, covA, freqs, ref_freq, beta, beam_vec, nside=nside, nside_new=nside_new, Nfreqs=len(freqs), beam_fwhm=beam_fwhm, lmax=lmax_beam, freq_beam = freq_beam)
        print("Done RHS")
        print("rhs ", np.min(rhs_eqn), np.max(rhs_eqn), flush=True)

        def lhs_op_amps(v): 
            return amps_lhs(v, gain, all_covN, covA, freqs, ref_freq, beta, beam_vec, nside=nside, nside_new=nside_new, Nfreqs=len(freqs), beam_fwhm=beam_fwhm, lmax=lmax_beam, freq_beam = freq_beam)

        lhs_matrix = linalg.LinearOperator(matvec=lhs_op_amps, shape=(int(npix), int(npix)))
        if op_alm == False:
            true_lhs = amps_lhs(true_s, operator @ mu_g , all_covN, covA, freqs, ref_freq, beta, beam_vec, nside=nside, nside_new=nside_new, Nfreqs=len(freqs), beam_fwhm=beam_fwhm, lmax = lmax_beam, freq_beam = freq_beam)
        else:
            true_lhs = amps_lhs(true_s, 1 + hp.alm2map(mu_g[:int(len_alms/2)] + 1j * mu_g[int(len_alms/2):], nside, lmax=lmax_gain), all_covN, covA, freqs, ref_freq, beta, beam_vec, nside=nside, nside_new=nside_new, Nfreqs=len(freqs), beam_fwhm=beam_fwhm, lmax = lmax_beam, freq_beam = freq_beam)
        
        lhs_val = amps_lhs(true_s, gain, all_covN, covA, freqs, ref_freq, beta, beam_vec, nside=nside, nside_new=nside_new, Nfreqs=len(freqs), beam_fwhm=beam_fwhm, lmax = lmax_beam, freq_beam = freq_beam)

        print("true lhs ", np.min(true_lhs), np.max(true_lhs), flush=True)
        print("lhs ", np.min(lhs_val), np.max(lhs_val), flush=True)

        print("min and max of lhs - rhs", np.min(lhs_val - rhs_eqn), np.max(lhs_val - rhs_eqn), flush=True)

        time_start = time.perf_counter()
        amps, info = linalg.cgs(A = lhs_matrix, b = rhs_eqn, maxiter=int(1e3))#, rtol=1e-9, atol=1e-1)#, x0 = s0[-1])#, rtol=1e-9)# atol=1e-1)#, M = np.linalg.pinv(lhs_matrix))
        time_end = time.perf_counter()
        print("amplitude: cgs took ", time_end - time_start, flush=True)
        print("Min and max diff of signal amplitude", np.min(amps - true_s), np.max(amps - true_s), flush=True)

        # time_start = time.perf_counter()
        # amps, info = linalg.bicgstab(A = lhs_matrix, b = rhs_eqn, maxiter=int(1e3))#, rtol=1e-9, atol=1e-1)#, x0 = s0[-1])#, rtol=1e-9)# atol=1e-1)#, M = np.linalg.pinv(lhs_matrix))
        # time_end = time.perf_counter()
        # print("amplitude: bicgstab took ", time_end - time_start, flush=True)
        # print("Min and max diff of signal amplitude", np.min(amps - true_s ), np.max(amps - true_s ), flush=True)

        # time_start = time.perf_counter()
        # amps, info = linalg.gmres(A = lhs_matrix, b = rhs_eqn, maxiter=int(1e3))#, rtol=1e-9, atol=1e-1)#, x0 = s0[-1])#, rtol=1e-9)# atol=1e-1)#, M = np.linalg.pinv(lhs_matrix))
        # time_end = time.perf_counter()
        # print(amps, info)
        # print("amplitude: gmres took ", time_end - time_start, flush=True)
        # print("Min and max diff of signal amplitude", np.min(amps - true_s ), np.max(amps - true_s ), flush=True)

        # time_start = time.perf_counter()
        # amps, info = linalg.minres(A = lhs_matrix, b = rhs_eqn, maxiter=int(1e3))#, rtol=1e-9, atol=1e-1)#, x0 = s0[-1])#, rtol=1e-9)# atol=1e-1)#, M = np.linalg.pinv(lhs_matrix))
        # time_end = time.perf_counter()
        # print(amps, info)
        # print("amplitude: minres took ", time_end - time_start, flush=True)
        # print("Min and max diff of signal amplitude", np.min(amps - true_s ), np.max(amps - true_s ), flush=True)

        # raise SystemExit

        if info>0:
            warnings.warn("Warning! Amplitude info: %i" %info)
            print("drawing amplitude again", flush=True)
            rhs_eqn = amps_rhs(all_data, gain, all_covN, covA, freqs, ref_freq, beta, mu, beam_vec, nside=nside, Nfreqs=len(freqs), lmax = lmax_beam, freq_beam = freq_beam)
            amps, info = linalg.cgs(A = lhs_matrix, b = rhs_eqn, maxiter=int(1e3), rtol=1e-9)#, M = np.linalg.pinv(lhs_matrix))
            time_end = time.perf_counter()
            print("amplitude: cgs took ", time_end - time_start, flush=True)

        
        print("Min and max diff of signal amplitude", np.min(amps - true_s), np.max(amps - true_s), flush=True)
        
        ########## combine everything to calculate posterior and chisq ##############
        all_model_ = calc_model_spectral(amps, freqs, gain, beta, len(data1), beam_fwhm, ref_freq, freq_beam, lmax_beam = lmax_beam, nside=nside, nside_new=nside_new)
        all_model_global[ii] = all_model_[0]
        all_model_haslam[ii] = all_model_[1]
        
        amps_npix = np.outer((freqs / ref_freq)**beta, amps)
        
        print("Min and max diff of Haslam signal amplitude", np.min(amps - true_s), np.max(amps - true_s), flush=True)

        for jj in range(len(freqs)):
            if jj<len(data1):
                covN = covN1
                
                ln_post[ii] += np.sum((- np.transpose(all_data[0][jj] - all_model_global[ii, jj]) / covN * (all_data[0][jj] - all_model_global[ii, jj]))) +  np.sum(- (amps_npix[jj] / covS * amps_npix[jj]))
                chisq[ii, jj, :hp.nside2npix(nside_new)] = ((all_data[0][jj] - all_model_global[ii, jj])**2/covN)

            else:
                covN = covN2
                ln_post[ii] += np.sum((- np.transpose(all_data[1] - all_model_haslam[ii]) / covN
                                                        * (all_data[1] - all_model_haslam[ii])
                                                       - (amps / covS * amps)))
                chisq[ii, jj] = ((all_data[1] - all_model_haslam[ii])**2/covN)

        
        all_s[ii] = amps.copy()
        print(ln_post[ii], np.mean(chisq[ii,:]))

        time_end = time.perf_counter()

        s0 = amps.copy()

        print("One iteration takes ", time_end - time_start_it, flush=True)    
        if ((ii>1) and (ii % 100 == 0)):
            print("Saving iteration %i-th" %ii, flush=True)
            if sig2 == sig3:
                np.savez(fout + "output_lnpost_a_s_partial_%i_nstart%i_new_alm%s-%i_fwhm%ideg_sig1%i_sig2%i_nside%i" %(ntimes, nstart, op_alm, ii, beam_deg, sig1, sig2, nside_new),
                 ln_post = ln_post[:ii], all_s= all_s[:ii], all_g = all_g[:ii], chisq=chisq[:ii],all_model_global = all_model_global[:ii], all_model_haslam = all_model_haslam[:ii])
            else:
                np.savez(fout + "output_lnpost_a_s_partial_%i_nstart%i_new_alm%s-%i_fwhm%ideg_sig1%i_sig2%i_sig3%i_nside%i" %(ntimes, nstart, op_alm, ii, beam_deg, sig1, sig2, sig3, nside_new),
                 ln_post = ln_post[:ii], all_s= all_s[:ii], all_g = all_g[:ii], chisq=chisq[:ii],all_model_global = all_model_global[:ii], all_model_haslam = all_model_haslam[:ii])

    return ln_post, all_s, all_g, chisq, all_model_global, all_model_haslam, covG, covA

if __name__ == '__main__':
    ln_post, all_s, all_g, chisq, all_model_global, all_model_haslam, covG, covA = gibbs_sampling(data1, data2, covN1, covN2, covG, ntimes,
                                                         operator, freqs, beta, true_s, ref_freq, indx_ref,
                                                        beam_fwhm, nstart = nstart, file_s0 = file_s0,
                                                        mu_g = gain.flatten(), op_alm = op_alm, nside = nside, nside_new = nside_new,
                                                        lmax_gain =lmax_gain, lmax_beam = lmax_beam, freq_beam=freq_beam)

    all_data = [data1, data2] #np.concatenate((data1, data2[np.newaxis, :]), axis=0)
    if sig2 == sig3:
        np.savez(fout + "output_lnpost_a_s_all_%i_nstart%i_new_alm%s_fwhm%ideg_sig1%i_sig2%i_nside%i" %(ntimes, nstart, op_alm, beam_deg, sig1, sig2, nside_new),
         ln_post = ln_post, all_s= all_s, all_g = all_g, chisq=chisq, data1 = data1, data2 = data2, all_model_global = all_model_global, all_model_haslam = all_model_haslam, covG= covG, covA=covA)
    else:
        np.savez(fout + "output_lnpost_a_s_all_%i_nstart%i_new_alm%s_fwhm%ideg_sig1%i_sig2%i_sig3%i_nside%i" %(ntimes, nstart, op_alm, beam_deg, sig1, sig2, sig3, nside_new),
         ln_post = ln_post, all_s= all_s, all_g = all_g, chisq=chisq, data1 = data1, data2 = data2, all_model_global = all_model_global, all_model_haslam = all_model_haslam, covG= covG, covA=covA)
    print(np.mean(all_g, axis=0))
