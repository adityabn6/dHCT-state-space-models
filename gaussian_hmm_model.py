#!/bin/python

from helpers import *

import numpy as np
#from hmmlearn import hmm
# needs GitHub version of PyHHMM (commit 6c2eae1 fixes an issue with seaborn compatibility)
import pyhhmm.utils
import csv
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib
import multiprocessing

# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_variational_inference.html
# https://github.com/fmorenopino/heterogeneoushmm - could allow us to address the missing values more directly
# https://pmc.ncbi.nlm.nih.gov/articles/PMC13186999/
# https://deepblue.lib.umich.edu/data/concern/data_sets/ht24wk394?locale=en

if __name__ == '__main__':
    # load and normalize per-patient data
    max_days = update_max_day("../data/daily_hr.csv", cohort=["Patients","Caregivers"])
    max_days = update_max_day("../data/daily_activity.csv", max_post_txp_day=max_days, cohort=["Patients","Caregivers"])
    max_days = update_max_day("../data/daily_steps.csv", max_post_txp_day=max_days, cohort=["Patients","Caregivers"])
    max_days = update_max_day("../data/mood.csv", max_post_txp_day=max_days, cohort=["Patients","Caregivers"])
    max_days = update_max_day("../data/sleep_stages.csv", max_post_txp_day=max_days, cohort=["Patients","Caregivers"])
    #max_days = update_max_day("../data/temperature.csv", max_post_txp_day=max_days, patient_key="id", dft_key="dft", group_key=None)
    dataset = init_data(max_days)

    load_update_data_dict_sparse("../data/daily_hr.csv", "mean_hr", dataset)
    load_update_data_dict_sparse("../data/daily_activity.csv", "percent_active", dataset)
    load_update_data_dict_sparse("../data/daily_steps.csv", "mean_steps_per_minute", dataset)
    load_update_data_dict_sparse("../data/mood.csv", "MOOD", dataset)
    load_update_data_dict_sparse("../data/sleep_stages.csv", "sleep_duration", dataset)
    #load_update_data_dict_sparse("../data/temperature.csv", "temp_f", dataset, patient_key="id", dft_key="dft")
    #temperature to celsius
    #copy_data_to_new_key(dataset, "temp_f", "temp_c")
    #apply_function_by_patient(dataset, "temp_c", lambda x: (x - 32) / 1.8)

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

    #log-transform sleep
    copy_data_to_new_key(dataset, "sleep_duration", "log_sleep_duration")
    apply_function_by_patient(dataset, "log_sleep_duration", lambda x: math.log(x))
    sleep_duration_flat = extract_by_key(dataset, "log_sleep_duration")
    f = qqplot_norm(sleep_duration_flat)
    f.suptitle("Log (sleep duration)")

    #temp_c_flat = extract_by_key(dataset, "temp_c")
    #f = qqplot_norm(temp_c_flat)
    #f.suptitle("Temperature")

    #zero-center HR
    copy_data_to_new_key(dataset, "mean_hr", "zero_centered_mean_hr")
    standardize_by_patient_and_key(dataset, "zero_centered_mean_hr", mean=0)
    zero_centered_mean_hr_flat = extract_by_key(dataset, "zero_centered_mean_hr")
    f = qqplot_norm(zero_centered_mean_hr_flat)
    f.suptitle("Zero-centered daily HR")

    #zero-center steps
    copy_data_to_new_key(dataset, "mean_steps_per_minute", "zero_centered_mean_steps_per_minute")
    standardize_by_patient_and_key(dataset, "zero_centered_mean_steps_per_minute", mean=0)
    zero_centered_step_vals_flat = extract_by_key(dataset, "zero_centered_mean_steps_per_minute")
    f = qqplot_norm(zero_centered_step_vals_flat)
    f.suptitle("Zero-centered mean steps per minute")

    plt.show(block=True)

    keys_to_include = ["zero_centered_mean_hr", "zero_centered_mean_steps_per_minute","MOOD","log_sleep_duration"]
    num_features = len(keys_to_include)
    #remove patient-days with no feature data
    dataset_no_na_days = strip_na_days(dataset)
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

    #add clinical annotation: read in outcome files (like readmission and outcome) into sparse map
    clinical_data = {}
    load_update_clinical_outcome("../data/infections.csv","date_culture_drawn",["culture_source","infection_type","infection_name"], clinical_data)
    load_update_clinical_outcome("../data/readmissions.csv","date_admit",["admission_reason"], clinical_data)

    #add state results to dataset
    states_inferred = optimal_bic_model.predict(sequence_data)
    for i in range(0, len(patients_sorted)):
        current_state_idx = 0
        patient = patients_sorted[i]
        days_sorted = [x for x in dataset[patient].keys()]
        days_sorted.sort()
        for j in range(0, len(days_sorted)):
            day = days_sorted[j]
            if day in dataset_no_na_days[patient]:
                dataset[patient][day]["state"] = states_inferred[i][current_state_idx]
                current_state_idx = current_state_idx + 1
            else:
                dataset[patient][day]["state"] = np.nan

    clinical_headers = ["culture_source","infection_type","infection_name","admission_reason"]
    data_headers = ["mean_hr","zero_centered_mean_hr","percent_active","mean_steps_per_minute","zero_centered_mean_steps_per_minute","MOOD","sleep_duration","log_sleep_duration","state"]
    output_headers = ["STUDY_PRTCPT_ID","DaysFromTransplant"] + data_headers + clinical_headers
    output_handle = open("../output/output.csv", 'w', newline='')
    output_writer = csv.DictWriter(output_handle, fieldnames=output_headers)
    output_writer.writeheader()
    for i in range(0, len(patients_sorted)):
        patient = patients_sorted[i]
        days_sorted = [x for x in dataset[patient].keys()]
        days_sorted.sort()
        for j in range(0, len(days_sorted)):
            day = days_sorted[j]
            current_row = {}
            for key in data_headers:
                current_row[key] = dataset[patient][day][key]
            if clinical_data.get(patient) is not None and clinical_data[patient].get(day) is not None:
                for clinical_key in clinical_headers:
                    if clinical_data[patient][day].get(clinical_key) is not None:
                        current_row[clinical_key] = clinical_data[patient][day][clinical_key]
                    else:
                        current_row[clinical_key] = ""
            else:
                for clinical_key in clinical_headers:
                    current_row[clinical_key] = ""
            current_row["STUDY_PRTCPT_ID"] = patient
            current_row["DaysFromTransplant"] = day
            output_writer.writerow(current_row)
    output_handle.close()

    print("Start probabilities:\n")
    print(optimal_bic_model.pi)
    print("\nTransition probabilities:\n")
    print(optimal_bic_model.A)
    print("\nMeans:\n")
    print(optimal_bic_model.means)
    print("\nCovariance matrices:\n")
    print(optimal_bic_model.covars)
    sys.stdout.flush()

    aic_list = []
    bic_list = []
    ll_list = []
    states_list = []
    for num_states in range(min_states, max_states + 1):
        if num_states in best_scores:
            dof = pyhhmm.utils.get_n_fit_scalars(best_models[num_states])
            aic = pyhhmm.utils.aic_hmm(best_scores[num_states], dof)
            bic = pyhhmm.utils.bic_hmm(best_scores[num_states], dof, total_observations)
            ll = best_scores[num_states]
            aic_list.append(aic)
            bic_list.append(bic)
            ll_list.append(ll)
            states_list.append(num_states)

    f = bic_graph(states_list, aic_list, bic_list, ll_list)

    plt.show()


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
