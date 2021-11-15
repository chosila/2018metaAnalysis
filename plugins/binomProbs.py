#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:20:17 2021

@author: chosila
"""

import numpy as np
import scipy
import scipy.stats as sci

def calc_pull(D_raw, R_list_raw, tol, optAB):
    intD = D_raw.sum()  ## Total number of data entries
    D_norm = D_raw / intD  ## Data entries per bin, normalized to 1.0

    nRef = len(R_list_raw)  ## Number of reference histograms
    intR_list = np.array([x.sum() for x in R_list_raw])  ## Total number of entries per ref hist
    intR_total = intR_list.sum()         ## Total number of entries summed across all ref hists
    R_norm_list = np.array([x*1/x.sum() for x in R_list_raw])  ## Array of normalized ref hists
    R_norm_avg = R_norm_list.mean(axis=0)  ## Average of normalized yields per bin in ref hists

    scaleTol = pow(1 + pow(R_list_raw * tol**2, 2), -0.5)  ## 2D array of scaling factors, nRef x nBins
    ## Multiply along only the first axis: see https://stackoverflow.com/questions/49435353/numpy-multiply-arbitrary-shape-array-along-first-axis
    intR_list_tol = (scaleTol.T * intR_list).T 
    R_list_raw_tol = R_list_raw * scaleTol  ## 2D array of scaled number of entries per ref hist bin


    ## Initialize arrays probHi and probLo with same dimensions as D_raw, values all 0
    probHi = np.zeros_like(D_raw)
    probLo = np.zeros_like(D_raw)

    for iRef in range(nRef):
        if optAB == 'A':  ## For approach 'A', all reference histograms receive equal weight
            wgt_AB = 1.0 / nRef
        if optAB == 'B':  ## For approach 'B', histograms weighted by number of entries        
            wgt_AB = 1.0 * intR_list[iRef] / intR_total

        probHi += wgt_AB * sci.betabinom.sf(D_raw-1, intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)
        probLo += wgt_AB * sci.betabinom.cdf(D_raw,  intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)

    ## For each bin, take the lower probability (i.e. the one that is < 50%)
    prob = probLo
    condition = probHi < probLo
    prob[condition] = probHi[condition]

    prob *= 2  ## Multiply by 2 for 2-sided probability    
    pull = calcPull(prob)
    
    return pull


def calcPull(prob):
    prob = np.clip(prob, a_min = np.power(0.1,16), a_max=None)## recalc prob so that it's not too small
    pull = np.sqrt(scipy.stats.chi2.ppf(1-prob,1)) ## convert to pull value
        
    return pull
