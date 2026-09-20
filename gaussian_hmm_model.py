#!/bin/python

from data_loader import load_data
from helpers import *

import numpy as np
#from hmmlearn import hmm
# needs GitHub version of PyHHMM (commit 6c2eae1 fixes an issue with seaborn compatibility)
import pyhhmm.utils
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib
import multiprocessing

# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_variational_inference.html
# https://github.com/fmorenopino/heterogeneoushmm - could allow us to address the missing values more directly
# https://pmc.ncbi.nlm.nih.gov/articles/PMC13186999/
# https://deepblue.lib.umich.edu/data/concern/data_sets/ht24wk394?locale=en

if __name__ == '__main__':
    dataset = load_data()

    patients_sorted = [x for x in dataset.keys()]
    patients_sorted.sort()
    num_patients = len(dataset)
    total_observations = sum([len(x) for x in dataset.values()])

    hr_vals_flat = extract_by_key(dataset, "mean_hr")
    f = qqplot_norm(hr_vals_flat)
    f.suptitle("Daily average HR")

    step_vals_flat = extract_by_key(dataset, "mean_steps_per_minute")
    f = qqplot_norm(step_vals_flat)
    f.suptitle("Mean steps per minute")

    mood_flat = extract_by_key(dataset, "MOOD")
    f = qqplot_norm(mood_flat)
    f.suptitle("Mood scores")

    sleep_duration_flat = extract_by_key(dataset, "log_sleep_duration")
    f = qqplot_norm(sleep_duration_flat)
    f.suptitle("Log (sleep duration)")

    #temp_c_flat = extract_by_key(dataset, "temp_c")
    #f = qqplot_norm(temp_c_flat)
    #f.suptitle("Temperature")

    zero_centered_mean_hr_flat = extract_by_key(dataset, "zero_centered_mean_hr")
    f = qqplot_norm(zero_centered_mean_hr_flat)
    f.suptitle("Zero-centered daily HR")

    zero_centered_step_vals_flat = extract_by_key(dataset, "zero_centered_mean_steps_per_minute")
    f = qqplot_norm(zero_centered_step_vals_flat)
    f.suptitle("Zero-centered mean steps per minute")

    plt.show(block=True)

    keys_to_include = ["zero_centered_mean_hr", "zero_centered_mean_steps_per_minute","MOOD","log_sleep_duration"]
    num_features = len(keys_to_include)
    #remove patient-days with less than two observations
    dataset_no_na_days = filter_minimum_obs_days(dataset, minimum_obs=2)
    sequence_data = package_observations_for_model(dataset_no_na_days, keys_to_include, patient_ordering = patients_sorted)

    # number of models we try at each number of states
    num_inits = 3
    # range of states to try
    min_states = 2
    max_states = 6

    num_processes = 6

    pool = multiprocessing.Pool(processes=num_processes)
    model_states_to_run = [x for x in range(min_states, max_states +1)] * num_inits
    models = [pool.apply_async(learn_model, (num_states, num_features, sequence_data, )) for num_states in model_states_to_run]
    pool.close()
    pool.join()

    try:
        os.mkdir("../output")
    except FileExistsError:
        pass

    best_scores = {}
    best_models = {}
    model_idx_by_state = {}
    for res in models:
        em = res.get()[0]
        if em is not None:
            num_states = em.n_states
            ll = res.get()[1]
            if best_models.get(num_states) is None or best_scores[num_states] < ll:
                best_models[num_states] = em
                best_scores[num_states] = ll
            if num_states not in model_idx_by_state:
                model_idx_by_state[num_states] = 1
            else:
                model_idx_by_state[num_states] = model_idx_by_state[num_states] + 1
            dof = pyhhmm.utils.get_n_fit_scalars(em)
            bic = pyhhmm.utils.bic_hmm(ll, dof, total_observations)
            filename = "../output/model_" + str(num_states) + "s_" + str(model_idx_by_state[num_states]) + ".pkl"
            print("Model for " + str(num_states) + " states, number " + str(model_idx_by_state[num_states]) + ": BIC " + str(bic))
            save_model(em, filename)

    optimal_bic_model = None
    optimal_bic = None
    for num_states in range(min_states, max_states + 1):
        if num_states in best_scores:
            dof = pyhhmm.utils.get_n_fit_scalars(best_models[num_states])
            bic = pyhhmm.utils.bic_hmm(best_scores[num_states], dof, total_observations)
            print(str(num_states) + " states: BIC " + str(bic))
            if optimal_bic_model is None or optimal_bic > bic:
                optimal_bic_model = best_models[num_states]
                optimal_bic = bic
        else:
            print("No model found for " + str(num_states) + " states.")

# inferring a resting heart rate? (for a patient, HR when steps are 0 and activity 1) - lab has data for this already
#   maybe normalize based on it (could be proxy for intrinsic SA nodal variability)
# try running this overnight with a higher max_states to get a more extensive BIC survey
# add sleep data to model - use stages (not classic) which is a more reliable algorithm from FitBit
# try learning model on patients with or without GVHD alone; if models look very different that would suggest something can be learned
# also try this for caregivers as a control
# if switching to PyHMM for missing value support, would need some QC (min observations, max % missing values)
# EECS 448
# https://atlas.ai.umich.edu/
# drop temperature; most of it was obtained during admissions
# try substituting sleep (sleep_duration) for mood
# try more granular data with e.g. 15-min bins
#   filter for bins where heart rate data is actually present 50% of the time
#   also bin steps; if missing row then assume steps = 0
#   include binary sleep or no sleep
