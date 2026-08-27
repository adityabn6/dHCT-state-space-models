#!/bin/python

import numpy as np
from hmmlearn import hmm
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib
import scipy.stats
import multiprocessing
from datetime import datetime, timedelta

# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_variational_inference.html
# https://pmc.ncbi.nlm.nih.gov/articles/PMC13186999/
# https://deepblue.lib.umich.edu/data/concern/data_sets/ht24wk394?locale=en

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
    proc_data = np.sort(data.flatten())
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
            #make sure we have a record for this patient-day
            if data.get(patient) is not None and data[patient].get(day) is not None:
                val = float(row[key])
                new_data[patient][day][key] = val
    file_handle.close()
    return new_data

#based on number of entires for a patient, flatten the list of data points into a single array
#if zero_center set to True, normalize for a mean of zero
def flatten_by_patient(data, sample_lengths, zero_center=False):
    flattened = []
    num_patients = len(sample_lengths)
    for i in range(0, num_patients):
        offset = 0
        if zero_center:
            offset = np.mean(data[i])
        for j in range(0, sample_lengths[i]):
            flattened.append(data[i][j] - offset)
    flattened = np.array(flattened)
    return flattened

#this updates data in place
def load_update_clinical_outcome(filepath, day_key, data_keys, data):
    #data -> {patient} -> {day post-txp} -> {features}
    file_handle = open(filepath)
    file_reader = csv.DictReader(file_handle, delimiter=',')
    for row in file_reader:
        patient = row["STUDY_PRTCPT_ID"]
        try:
            day = int(row[day_key])
            if data.get(patient) is None:
                data[patient] = {}
            if data[patient].get(day) is None:
                data[patient][day] = {}
            for key in data_keys:
                data[patient][day][key] = row[key]
        except ValueError:
            pass
    file_handle.close()

def learn_model(num_states, num_features, sequence_data, sample_lengths):
    start_time = datetime.now()
    em = hmm.GaussianHMM(num_states, n_iter=1000, covariance_type="full",implementation="scaling",tol=1e-6,verbose=False)
    em.n_features = num_features
    em.fit(sequence_data, sample_lengths)
    end_time = datetime.now()
    run_time = end_time - start_time
    print("Learned model for " + str(num_states) + " states. Time: " + str(run_time.total_seconds()) + " sec.", file=sys.stderr)
    return em

def extract_paired_dist(means, cov, i, j):
    pair_means = np.transpose(np.vstack([means[:,i],means[:,j]]))
    pair_cov = np.array([x[np.ix_([i,j],[i,j])] for x in cov])
    return pair_means, pair_cov

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

if __name__ == '__main__':
    # load and normalize per-patient data
    dataset = None
    dataset = load_update_data_dict("../data/daily_hr.csv", "mean_hr", dataset, cohort=["Patients","Caregivers"])
    dataset = load_update_data_dict("../data/daily_activity.csv", "percent_active", dataset, cohort=["Patients","Caregivers"])
    #for this, how to normalize? some people will have varying activity at baseline
    dataset = load_update_data_dict_fill("../data/daily_steps.csv", "mean_steps_per_minute", dataset, cohort=["Patients","Caregivers"])
    #lots of missing data points, maybe throw in something else first
    #also many instances of multiple survey results on the same day
    #dataset = load_update_data_dict("../data/mood.csv", "MOOD", dataset)

    patients_sorted = [x for x in dataset.keys()]
    patients_sorted.sort()
    num_patients = len(dataset)

    sample_lengths = []
    hr_vals = []
    activity_vals = []
    step_vals = []
    mood_vals = []
    for patient in patients_sorted:
        sample_lengths.append(len(dataset[patient]))
        current_hr_vals = []
        current_activity_vals = []
        current_step_vals = []
        #current_mood_vals = []
        days_sorted = [x for x in dataset[patient].keys()]
        days_sorted.sort()
        for day in days_sorted:
            current_hr_vals.append(dataset[patient][day]["mean_hr"])
            current_activity_vals.append(dataset[patient][day]["percent_active"])
            current_step_vals.append(dataset[patient][day]["mean_steps_per_minute"])
            #current_mood_vals.append(dataset[patient][day]["MOOD"])
        hr_vals.append(current_hr_vals)
        activity_vals.append(current_activity_vals)
        step_vals.append(current_step_vals)
        #mood_vals.append(current_mood_vals)

    zero_centered_hr_vals_flat = flatten_by_patient(hr_vals, sample_lengths, zero_center=True)
    f = qqplot_norm(zero_centered_hr_vals_flat)
    f.suptitle("zero_centered_hr_vals_flat")

    activity_vals_flat = flatten_by_patient(activity_vals, sample_lengths)
    f = qqplot_norm(activity_vals_flat)
    f.suptitle("activity_vals_flat")
    activity_vals_flat_log_transform = np.log([x+1 for x in activity_vals_flat])
    f = qqplot_norm(activity_vals_flat_log_transform)
    f.suptitle("activity_vals_flat_log_transform")

    zero_centered_step_vals_flat = flatten_by_patient(step_vals, sample_lengths, zero_center=True)
    f = qqplot_norm(zero_centered_step_vals_flat)
    f.suptitle("zero_centered_step_vals_flat")

    plt.show(block=True)

    sequence_data = np.transpose(np.vstack([zero_centered_hr_vals_flat, zero_centered_step_vals_flat]))
    num_features = np.shape(sequence_data)[1]

    # number of models we try at each number of states
    num_inits = 10
    # range of states to try
    min_states = 2
    max_states = 12

    num_processes = 6

    pool = multiprocessing.Pool(processes=num_processes)
    model_states_to_run = [x for x in range(min_states, max_states +1)] * num_inits
    models = [pool.apply_async(learn_model, (num_states, num_features, sequence_data, sample_lengths,)) for num_states in model_states_to_run]
    pool.close()
    pool.join()

    best_scores = {}
    best_models = {}
    for res in models:
        em = res.get()
        num_states = em.n_components
        ll = em.monitor_.history[-1]
        if best_models.get(num_states) is None or best_scores[num_states] < ll:
            best_models[num_states] = em
            best_scores[num_states] = ll

    optimal_bic_model = None
    optimal_bic = None
    for num_states in range(min_states, max_states + 1):
        bic = best_models[num_states].bic(sequence_data, sample_lengths)
        print(str(num_states) + " states: BIC " + str(bic))
        if optimal_bic_model is None or optimal_bic > bic:
            optimal_bic_model = best_models[num_states]
            optimal_bic = bic

    #add clinical annotation: read in outcome files (like readmission and outcome) into sparse map
    clinical_data = {}
    load_update_clinical_outcome("../data/infections.csv","date_culture_drawn",["culture_source","infection_type","infection_name"], clinical_data)
    load_update_clinical_outcome("../data/readmissions.csv","date_admit",["admission_reason"], clinical_data)

    states_inferred = optimal_bic_model.predict(sequence_data, sample_lengths)
    clinical_headers = ["culture_source","infection_type","infection_name","admission_reason"]
    output_headers = ["STUDY_PRTCPT_ID","DaysFromTransplant","mean_hr","percent_active","mean_steps_per_minute","state"] + clinical_headers
    output_handle = open("../output.csv", 'w', newline='')
    output_writer = csv.DictWriter(output_handle, fieldnames=output_headers)
    output_writer.writeheader()
    current_state_idx = 0
    for patient in patients_sorted:
        days_sorted = [x for x in dataset[patient].keys()]
        days_sorted.sort()
        for day in days_sorted:
            dataset[patient][day]["state"] = states_inferred[current_state_idx]
            current_row = dataset[patient][day]
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
            current_state_idx = current_state_idx + 1
    output_handle.close()

    print("Start probabilities:\n")
    print(optimal_bic_model.startprob_)
    print("\nTransition probabilities:\n")
    print(optimal_bic_model.transmat_)
    print("\nMeans:\n")
    print(optimal_bic_model.means_)
    print("\nCovariance matrices:\n")
    print(optimal_bic_model.covars_)
    sys.stdout.flush()

    aic_list = []
    bic_list = []
    ll_list = []
    for num_states in range(min_states, max_states + 1):
        aic_list.append(best_models[num_states].aic(sequence_data, sample_lengths))
        bic_list.append(best_models[num_states].bic(sequence_data, sample_lengths))
        ll_list.append(best_models[num_states].score(sequence_data, sample_lengths))

    f = bic_graph(range(min_states, max_states + 1), aic_list, bic_list, ll_list)

    # any 3d plot libraries? would be helpful to see the joint probability dists for pairs of features
    f = gaussian_hinton_diagram(
        optimal_bic_model.startprob_,
        optimal_bic_model.transmat_,
        optimal_bic_model.means_[:,0].ravel(),
        [x[0][0] for x in optimal_bic_model.covars_],
        infer_hidden=False,
    )
    f.suptitle("Expectation-Maximization Solution, Feature 0", size=16)

    f = gaussian_hinton_diagram(
        optimal_bic_model.startprob_,
        optimal_bic_model.transmat_,
        optimal_bic_model.means_[:,1].ravel(),
        [x[1][1] for x in optimal_bic_model.covars_],
        infer_hidden=False,
    )
    f.suptitle("Expectation-Maximization Solution, Feature 1", size=16)

    feature_0_1_means, feature_0_1_covars = extract_paired_dist(optimal_bic_model.means_, optimal_bic_model.covars_, 0, 1)
    f = multi_gaussian_3d_plot(feature_0_1_means, feature_0_1_covars, xlabel="Feature 0", ylabel="Feature 1")

    plt.show()


# inferring a resting heart rate? (for a patient, HR when steps are 0 and activity 1) - lab has data for this already
#   maybe normalize based on it (could be proxy for intrinsic SA nodal variability)
# try running this overnight with a higher max_states to get a more extensive BIC survey
# add sleep data to model - use stages (not classic) which is a more reliable algorithm from FitBit
# try learning model on patients with or without GVHD alone; if models look very different that would suggest something can be learned
# also try this for caregivers as a control
