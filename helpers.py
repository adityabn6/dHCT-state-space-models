#!/bin/python

import numpy as np
#from hmmlearn import hmm
# needs GitHub version of PyHHMM (commit 6c2eae1 fixes an issue with seaborn compatibility)
from pyhhmm.gaussian import GaussianHMM
import pickle
import pyhhmm.utils
import csv
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib
import scipy.stats
import multiprocessing
from datetime import datetime, timedelta

def gaussian_hinton_diagram(startprob, transmat, means, variances, vmin=0, vmax=1, infer_hidden=True):
    num_states = transmat.shape[0]
    f = plt.figure(figsize=(3*(num_states), 2*num_states))
    grid = gs.GridSpec(3, 3)
    ax = f.add_subplot(grid[0, 0])
    ax.imshow(startprob[None, :], vmin=vmin, vmax=vmax)
    ax.set_title("Initial Probabilities", size=14)
    ax = f.add_subplot(grid[1:, 0])
    ax.imshow(transmat, vmin=vmin, vmax=vmax)
    ax.set_title("Transition Probabilities", size=14)
    ax = f.add_subplot(grid[1:, 1:])
    for i in range(num_states):
        keep = True
        if infer_hidden:
            if np.all(np.abs(transmat[i] - transmat[i][0]) < 1e-4):
                keep = False
        if keep:
            s_min = means[i] - 10 * variances[i]
            s_max = means[i] + 10 * variances[i]
            xx = np.arange(s_min, s_max, (s_max - s_min) / 1000)
            norm = scipy.stats.norm(means[i], np.sqrt(variances[i]))
            yy = norm.pdf(xx)
            keep = yy > .01
            ax.plot(xx[keep], yy[keep], label="State: {}".format(i))
    ax.set_title("Emissions Probabilities", size=14)
    ax.legend(loc="best")
    f.tight_layout()
    return f

def multi_gaussian_3d_plot(means, cov, xlabel="Feature 0", ylabel="Feature 1"):
    num_states = means.shape[0]
    models = []
    s_min_x = None
    s_max_x = None
    s_min_y = None
    s_max_y = None
    for i in range(num_states):
        model = scipy.stats.multivariate_normal(means[i], cov[i], allow_singular=True)
        models.append(model)
        x_min = means[i][0] - 20 #(10 * cov[i][0][0])
        x_max = means[i][0] + 20 #(10 * cov[i][0][0])
        y_min = means[i][1] - 20 #(10 * cov[i][1][1])
        y_max = means[i][1] + 20 #(10 * cov[i][1][1])
        if s_min_x is None or x_min < s_min_x:
            s_min_x = x_min
        if s_max_x is None or x_max > s_max_x:
            s_max_x = x_max
        if s_min_y is None or y_min < s_min_y:
            s_min_y = y_min
        if s_max_y is None or y_max > s_max_y:
            s_max_y = y_max
    x_points = np.linspace(s_min_x, s_max_x, 100)
    y_points = np.linspace(s_min_y, s_max_y, 100)
    X, Y = np.meshgrid(x_points, y_points)
    z_val = np.zeros(len(y_points) * len(x_points)).reshape(len(y_points), len(x_points))
    z_max = 0
    color_val = np.zeros(len(y_points) * len(x_points)).reshape(len(y_points), len(x_points))
    for i in range(0, len(y_points)):
        for j in range(0, len(x_points)):
            x = X[i,j]
            y = Y[i,j]
            max_val = 0
            max_idx = None
            for k in range(0, num_states):
                val = models[k].pdf([x, y])
                if val > max_val:
                    max_val = val
                    max_idx = k
            if max_val > 0.0001:
                color_val[i,j] = max_idx + 1
            else:
                color_val[i,j] = 0
            z_val[i,j] = max_val
            if max_val > z_max:
                z_max = max_val
    norm = plt.Normalize(0, num_states)
    cm = matplotlib.colormaps['tab20']
    rgba_colors = cm(norm(color_val))
    f = plt.figure(figsize=(10,7))
    ax = f.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, z_val, facecolors=rgba_colors, linewidth=0, antialiased=True, shade=False)
    ax.set_xlim(s_min_x, s_max_x)
    ax.set_ylim(s_min_y, s_max_y)
    ax.set_zlim(0, z_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    mappable.set_array(color_val)
    f.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='State')
    return f

def bic_graph(ns, aic, bic, lls):
    f, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    f.tight_layout()
    return f

def qqplot_norm(data):
    proc_data = np.sort(np.array(data).flatten())
    n = len(proc_data)
    probs = np.linspace(0.5 / n, 1 - (0.5 / n), n)
    #data_quantiles = np.quantile(proc_data, probs)
    theoretical_quantiles = scipy.stats.norm.ppf(probs)
    f, ax = plt.subplots()
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.scatter(theoretical_quantiles, proc_data)
    return f

#for a specific file, column key, and previous data set, return a new data set with desired data from this file while dropping entries that are not in this file
def load_update_data_dict(filepath, key, data, cohort=["Patients"]):
    #data -> {patient} -> {day post-txp} -> {features}
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    new_data = {}
    for row in file_reader:
        if row["Group"] in cohort:
            patient = row["STUDY_PRTCPT_ID"]
            day = int(row["DaysFromTransplant"])
            if day >= 0:
                #if there's already existing data, ensure that other data is present for this patient and day
                #eventually would want to look into missing value imputation upstream of this script
                if data is not None and (data.get(patient) is None or data[patient].get(day) is None):
                    continue
                if patient not in new_data:
                    new_data[patient] = {}
                new_data[patient][day] = {}
                if data is not None:
                    for k, v in data[patient][day].items():
                        new_data[patient][day][k] = v
                val = float(row[key])
                new_data[patient][day][key] = val
    file_handle.close()
    return new_data

#for a specific file, column key, and previous data set, return a new data set with desired data from this file while entries that are not in this file are set to fill=
def load_update_data_dict_fill(filepath, key, data, fill=0, cohort=["Patients"]):
    new_data = {}
    #pre-fill the array
    for patient in data:
        for day in data[patient]:
            if patient not in new_data:
                new_data[patient] = {}
            new_data[patient][day] = {}
            for k, v in data[patient][day].items():
                new_data[patient][day][k] = v
            new_data[patient][day][key] = fill
    #then overwrite pre-filled values with whatever's in the input file
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    for row in file_reader:
        if row["Group"] in cohort:
            patient = row["STUDY_PRTCPT_ID"]
            day = int(row["DaysFromTransplant"])
            if day >= 0:
                #make sure we have a record for this patient-day
                if data.get(patient) is not None and data[patient].get(day) is not None:
                    val = float(row[key])
                    new_data[patient][day][key] = val
    file_handle.close()
    return new_data

#pull out all values in a data set for a particular key into an array
#for use when aggregate metrics are needed, e.g. QC, Q-Q plots, etc.
#no guarantees are made regarding patient order
#if patients is specified, limit to those patients
def extract_by_key(data, key, skip_nan=True, patients=None):
    extract = []
    patients_to_include = data.keys()
    if patients is not None:
        patients_to_include = patients
    for patient in patients_to_include:
        for day in data[patient]:
            val = data[patient][day][key]
            if not (skip_nan and np.isnan(val)):
                extract.append(val)
    return extract

def copy_data_to_new_key(data, key, new_key):
    for patient in data:
        for day in data[patient]:
            data[patient][day][new_key] = data[patient][day][key]

#standardize data per-patient based on desired statistical parameters
#to get z-scores, set mean=0 and sd=1
#if mean or SD is none, they are ignored
def standardize_by_patient_and_key(data, key, mean=None, std=None):
    for patient in data:
        mean_correct_factor = 0
        stdev_correct_factor = 1.0
        vals = np.array(extract_by_key(data, key, patients=[patient]))
        if len(vals) > 0:
            if mean is not None:
                mean_correct_factor = np.mean(vals) - mean
            if std is not None:
                stdev_correct_factor = np.std(vals) / std
            for day in data[patient]:
                data[patient][day][key] = (data[patient][day][key] - mean_correct_factor) / stdev_correct_factor

#apply a function to every value in the dataset for a specified key
def apply_function_by_patient(data, key, func):
    for patient in data:
        for day in data[patient]:
            if data[patient][day][key] is not np.nan:
                data[patient][day][key] = func(data[patient][day][key])

#this updates data in place
def load_update_clinical_outcome(filepath, day_key, data_keys, data):
    #data -> {patient} -> {day post-txp} -> {features}
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    for row in file_reader:
        patient = row["STUDY_PRTCPT_ID"]
        try:
            day = int(row[day_key])
            if day >= 0:
                if data.get(patient) is None:
                    data[patient] = {}
                if data[patient].get(day) is None:
                    data[patient][day] = {}
                for key in data_keys:
                    data[patient][day][key] = row[key]
        except ValueError:
            pass
    file_handle.close()

#find the maximum post-txp day for each patient in an input CSV
#if already provided a dict of post-txp days will return a dict of the maximum day between either the input CSV or the input
def update_max_day(filepath, max_post_txp_day = None, cohort=["Patients"], patient_key="STUDY_PRTCPT_ID", dft_key="DaysFromTransplant", group_key="Group"):
    new_max_post_txp_day = {}
    if max_post_txp_day is not None:
        for patient, day in max_post_txp_day.items():
            new_max_post_txp_day[patient] = day
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    for row in file_reader:
        if group_key is None or row[group_key] in cohort:
            patient = row[patient_key]
            if patient not in new_max_post_txp_day:
                new_max_post_txp_day[patient] = 0
            day = int(row[dft_key])
            if day > new_max_post_txp_day[patient]:
                new_max_post_txp_day[patient] = day
    file_handle.close()
    return new_max_post_txp_day

def init_data(max_post_txp_day):
    data = {}
    for k, v in max_post_txp_day.items():
        data[k] = {}
        for i in range(0, v+1):
            data[k][i] = {}
    return data

def load_update_data_dict_sparse(filepath, key, data, fill=np.nan, aggregate_func = lambda x: sum(x) / len(x), patient_key="STUDY_PRTCPT_ID", dft_key="DaysFromTransplant"):
    #pre-fill the dict
    for patient in data:
        for day in data[patient]:
            data[patient][day][key] = fill
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    sparse_data = {}
    for row in file_reader:
        patient = row[patient_key]
        day = int(row[dft_key])
        if day >=0:
            if patient not in sparse_data:
                sparse_data[patient] = {}
            if day not in sparse_data[patient]:
                sparse_data[patient][day] = []
            val = float(row[key])
            sparse_data[patient][day].append(val)
    file_handle.close()
    for patient in sparse_data:
        #don't register new patients
        if patient in data:
            for day in sparse_data[patient]:
                val = aggregate_func(sparse_data[patient][day])
                data[patient][day][key] = val

#generate a copy of a dataset filtered for patient-days with at least this many non-NA observations
def filter_minimum_obs_days(data, minimum_obs=1):
    new_data = {}
    for patient in data:
        new_patient_data = {}
        for day in data[patient]:
            obs_count = 0
            for k, v in data[patient][day].items():
                if not np.isnan(v):
                    obs_count = obs_count + 1
            if obs_count >= minimum_obs:
                new_patient_data[day] = {}
                for k, v in data[patient][day].items():
                    new_patient_data[day][k] = v
        if len(new_patient_data) > 0:
            new_data[patient] = new_patient_data
    return new_data

#generate an array of length num_patients containing (patient_days x features)
#can specify a patient ordering
def package_observations_for_model(data, keys, patient_ordering=None):
    output = []
    num_features = len(keys)
    patients_ordered = data.keys()
    if patient_ordering is not None:
        patients_ordered = patient_ordering
    for patient in patients_ordered:
        if patient in data:
            num_days = len(data[patient])
            feature_mat = np.zeros(num_days * num_features).reshape(num_days, num_features)
            day_idx = 0
            days_sorted = [x for x in data[patient]]
            days_sorted.sort()
            for day in days_sorted:
                for i in range(0, num_features):
                    feature_mat[day_idx, i] = data[patient][day][keys[i]]
                day_idx = day_idx + 1
            output.append(feature_mat)
    return(output)

def learn_model(num_states, num_features, sequence_data, n_iter=100):
    start_time = datetime.now()
    em = None
    log_likelihood = None
    print("PID " + str(os.getpid()) + ": Learning model for " + str(num_states) + " states.", file=sys.stderr)
    try:
        trained_em = GaussianHMM(n_states=num_states, n_emissions=num_features, covariance_type="diagonal", verbose=True)
        trained_em, trained_log_likelihood = trained_em.train(sequence_data, n_init=1, n_iter=n_iter, conv_thresh=0.001, conv_iter=5, print_every=10)
        em = trained_em
        log_likelihood = trained_log_likelihood
        end_time = datetime.now()
        run_time = end_time - start_time
        print("PID " + str(os.getpid()) + ": Learned model for " + str(num_states) + " states. Time: " + str(run_time.total_seconds()) + " sec.", file=sys.stderr)
    except ValueError as e:
        end_time = datetime.now()
        run_time = end_time - start_time
        print("PID " + str(os.getpid()) + ": Failed to learn model for " + str(num_states) + " states. Time: " + str(run_time.total_seconds()) + " sec." + "Error: " + str(e), file=sys.stderr)
    return (em, log_likelihood)

def extract_paired_dist(means, cov, i, j):
    pair_means = np.transpose(np.vstack([means[:,i],means[:,j]]))
    pair_cov = np.array([x[np.ix_([i,j],[i,j])] for x in cov])
    return pair_means, pair_cov

#broken in PyHHMM
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
