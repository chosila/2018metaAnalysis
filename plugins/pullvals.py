import ROOT
from plugin_results import PluginResults
import numpy as np
import root_numpy
import uproot
import scipy
import scipy.stats

def comparators():
    return {
        'pull_values': pullvals
    }

def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=100000, norm_type='all',
             **kwargs):
    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    ## if ref hist is empty, don't include it
    ref_hists_list = [x for x in histpair.ref_hists_list if x.values.sum() > 0]
    nRef = len(ref_hists_list)


    ## check to make sure histogram is TH2    
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None


    data_raw = data_hist.values
    ref_list_raw = np.array([x.values for x in ref_hists_list])
        
   
    ## num entries
    data_hist_Entries = np.sum(data_raw)
    ref_hist_Entries = ref_list_raw.mean(axis=0).sum()
    
    # Reject empty histograms
    is_good = data_hist_Entries != 0
        
    ## outside if because need ref_norm_avg for nBinsUsed
    ref_list_norm = np.array([x*1/x.sum() for x in ref_list_raw])
    ref_norm_avg = ref_list_norm.mean(axis=0)
    # do pull calculation only if data has entries
    if is_good and nRef > 0:
        ## calculate normalized ref and data arrays
        
        data_norm = data_raw*1/data_raw.sum()
        varR_norm = ref_list_norm.var(axis=0)
        sumR = ref_list_raw.sum(axis=0)
        
        
    
        pulls = pull(data_norm, ref_norm_avg, data_hist_Entries, varR_norm, sumR, nRef)
    else: 
        pulls = np.zeros_like(data_raw)


        
        
    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_norm_avg, data_raw)) 
    chi2 = np.square(pulls).sum()
    max_pull = maxPullNorm(pulls, nBinsUsed).max()
    nBins = data_hist.numbins
    
    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries,
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : (pulls, data_hist.edges) 
        
    }
    
    artifacts = [pulls, 'data_text', 'ref_text']
    c = ROOT.TCanvas('c', 'Pull') 
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)
    
    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)
    
    
    
def pull(D_norm, R_norm_avg, intD, varR_norm, sumR, nRef):
    ## plan - declare array of case sumR_j == 0, then use it as an "out" array for np.divide
    ##        to fill in places where the case sumR_j != 0 
    
    out = D_norm * sumR / np.sqrt( 1 + sumR / intD )
    
    numerator = D_norm - R_norm_avg
    denom1 = np.divide(R_norm_avg / intD)
    denom2 = varR_norm
    denom3 = np.divide(R_norm_avg*R_norm_avg, sumR, where=sumR!=0)
    denom4 = 0.01*0.01*np.square(D_norm + R_norm_avg)
    denominator = np.sqrt( denom1 + denom2 + denom3 + denom4 )
    
    pull = np.divide( numerator, denominator, out=out, where=R_norm_avg!=0)
    
    return pull
    
    
    
    
def maxPullNorm(maxPull, nBinsUsed):
    probGood = 1-scipy.stats.chi2.cdf(np.power(maxPull,2),1) # will give the same result as ROOT.TMath.Prob
    ## probGood = ROOT.TMath.Prob(np.power(maxPull, 2), 1)
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = np.minimum(probBadNorm, 1 - np.power(0.1,16))
    return np.sqrt(scipy.stats.chi2.ppf(val,1))
    
    
