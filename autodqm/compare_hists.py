#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import uproot
#from autodqm import cfg
import cfg
#from autodqm.histpair import HistPair
from histpair import HistPair
#sys.path.insert(1, '/Users/si_sutantawibul1/Projects/metaAnalysis/plugins')
import numpy as np

def process(config_dir, subsystem,
            data_series, data_sample, data_run, data_path,
            ref_series, ref_sample, ref_run, ref_path,
            output_dir='./out/', plugin_dir='./plugins/'):
    

    # Ensure no graphs are drawn to screen and no root messages are sent to
    # terminal

    histpairs = compile_histpairs(config_dir, subsystem,
                                  data_series, data_sample, data_run, data_path,
                                  ref_series, ref_sample, ref_run, ref_path)

    hist_outputs = []

    comparator_funcs = load_comparators(plugin_dir)
    for hp in histpairs:
        try:
            comparators = [(c, comparator_funcs[c]) for c in hp.comparators]
        except KeyError as e:
            raise error("Comparator {} was not found.".format(str(e)))

        for comp_name, comparator in comparators:

            result_id = identifier(hp, comp_name)
            #pdf_path = '{}/pdfs/{}.pdf'.format(output_dir, result_id)
            #json_path = '{}/jsons/{}.json'.format(output_dir, result_id)
            #png_path = '{}/pngs/{}.png'.format(output_dir, result_id)

            #if True:#not os.path.isfile(json_path):
            results = comparator(hp, **hp.config)

            # Continue if no results
            if not results:
                continue


            # Make pdf
            # results.canvas.Update()
            #results.canvas.SaveAs(pdf_path)

            hists = list()
            for i in results.root_artifacts:
                if isinstance(i, np.ndarray):
                    hists.append(i)
                #if i.InheritsFrom('TH1'):
                #    hist.append(i)


            # Make png
            #subprocess.Popen(
            #    ['convert', '-density', '50', '-trim', '-fuzz', '1%', pdf_path, png_path])

            # Make json
            info = {
                'id': result_id,
                'name': hp.data_name,
                'comparator': comp_name,
                'display': results.show or hp.config.get('always_show', False),
                #'config': hp.config,
                #'results': results.info,
                'info': results.info,
                #'pdf_path': pdf_path,
                #'json_path': json_path,
                #'png_path': png_path,
                'artifacts': results.root_artifacts,
                'hists' : hists
            }
            # with open(json_path, 'w') as jf:
            #     json.dump(info, jf)


            hist_outputs.append(info)

    return hist_outputs


def compile_histpairs(config_dir, subsystem,
                      data_series, data_sample, data_run, data_path,
                      ref_series, ref_sample, ref_run, ref_path):

    config = cfg.get_subsystem(config_dir, subsystem)
    # Histogram details
    conf_list = config["hists"]
    main_gdir = config["main_gdir"]

    # ROOT files
    data_file = uproot.open(data_path)
    ref_file = uproot.open(ref_path)
    
    histPairs = []
    
    histlist = []

    for hconf in conf_list:
        # Get name of hist in root file
        h = str(hconf["path"].split("/")[-1])
        # Get parent directory of hist
        gdir = str(hconf["path"].split(h)[0])

        data_dirname = "{0}{1}".format(main_gdir.format(data_run), gdir)
        ref_dirname = "{0}{1}".format(main_gdir.format(ref_run), gdir)

        data_dir = data_file[data_dirname[:-1]]
        ref_dir = ref_file[ref_dirname[:-1]]

        
        if not data_dir:
            raise error(
                "Subsystem dir {0} not found in data root file".format(data_dirname))
        if not ref_dir:
            raise error(
                "Subsystem dir {0} not found in ref root file".format(ref_dirname))



        data_keys = data_dir.keys()
        ref_keys = ref_dir.keys()
        
        valid_names = []
    
        
        # Add existing histograms that match h
        if "*" not in h:
             if h in [str(keys)[0:-2] for keys in data_keys] and h in [str(keys)[0:-2] for keys in ref_keys]:
                 try:
                     data_hist = data_dir[h]
                     ref_hist = ref_dir[h]
                 except Exception as e:
                     continue
                 hPair = HistPair(hconf,
                                  data_series, data_sample, data_run, str(h), data_hist,
                                  ref_series, ref_sample, ref_run, str(h), ref_hist)
                 histPairs.append(hPair)
        else:
            # Check entire directory for files matching wildcard (Throw out wildcards with / in them as they are not plottable)
            for name in data_keys:
                if h.split("*")[0] in str(name) and name in ref_keys and not "<" in str(name):
                    if("/" not in name[:-2]):
                        try:
                            data_hist = data_dir[name[:-2]]
                            ref_hist = ref_dir[name[:-2]]
                        except Exception as e:
                            continue
                        hPair = HistPair(hconf,
                                         data_series, data_sample, data_run, str(name[:-2]), data_hist,
                                         ref_series, ref_sample, ref_run, str(name[:-2]), ref_hist)
                        histPairs.append(hPair)

    ## write out dqmhists to a file 
    

    return histPairs


def load_comparators(plugin_dir):
    """Load comparators from each python module in ADQM_PLUGINS."""
    comparators = dict()


    for modname in os.listdir(plugin_dir):
        print(modname)

    for modname in os.listdir(plugin_dir):
        if modname[0] == '_' or modname[-4:] == '.pyc' or modname[0] == '.':
            continue
        if modname[-3:] == '.py':
            modname = modname[:-3]
        try:
            sys.path.insert(1, '/home/chosila/Projects/metaAnalysis/plugins') 
            sys.path.append('/home/chosila/Projects/metaAnalysis/plugins')
            # mod = __import__(f"{modname}")
            if modname == 'ks':
                import ks as mod
            elif modname == 'pullvals':
                import pullvals as mod
            else:
                continue


            new_comps = mod.comparators()

        except AttributeError:
            raise error(
                "Plugin {} does not have a comparators() function.".format(mod))
        comparators.update(new_comps)

    return comparators


def identifier(hp, comparator_name):
    """Return a `hashed` identifier for the histpair"""
    data_id = "DATA-{}-{}-{}".format(hp.data_series,
                                     hp.data_sample, hp.data_run)
    ref_id = "REF-{}-{}-{}".format(hp.ref_series, hp.ref_sample, hp.ref_run)
    if hp.data_name == hp.ref_name:
        name_id = hp.data_name
    else:
        name_id = "DATANAME-{}_REFNAME-{}".format(hp.data_name, hp.ref_name)
    comp_id = "COMP-{}".format(comparator_name)

    hash_snippet = str(hash(hp))[-5:]

    idname = "{}_{}_{}_{}_{}".format(
        data_id, ref_id, name_id, comp_id, hash_snippet)
    return idname


class error(Exception):
    pass
