#!/bin/python

from data_loader import load_data
from helpers import *

import numpy as np
import pyhhmm.utils
import sys
import matplotlib.pyplot as plt

# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_variational_inference.html
# https://github.com/fmorenopino/heterogeneoushmm - could allow us to address the missing values more directly
# https://pmc.ncbi.nlm.nih.gov/articles/PMC13186999/
# https://deepblue.lib.umich.edu/data/concern/data_sets/ht24wk394?locale=en

if __name__ == '__main__':
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    model = pyhhmm.utils.load_model(model_path)

    dataset = load_data()
    patients_sorted = [x for x in dataset.keys()]
    patients_sorted.sort()
    keys_to_include = ["zero_centered_mean_hr", "zero_centered_mean_steps_per_minute","MOOD","log_sleep_duration"]
    #remove patient-days with no feature data
    dataset_no_na_days = strip_na_days(dataset)
    sequence_data = package_observations_for_model(dataset_no_na_days, keys_to_include, patient_ordering = patients_sorted)

    #add clinical annotation: read in outcome files (like readmission and outcome) into sparse map
    clinical_data = {}
    load_update_clinical_outcome("../data/infections.csv","date_culture_drawn",["culture_source","infection_type","infection_name"], clinical_data)
    load_update_clinical_outcome("../data/readmissions.csv","date_admit",["admission_reason"], clinical_data)

    #add state results to dataset
    states_inferred = model.predict(sequence_data)
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
    output_handle = open(output_path, 'w', newline='')
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
