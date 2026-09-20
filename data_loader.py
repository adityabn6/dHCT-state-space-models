#!/bin/python

from helpers import update_max_day, init_data, load_update_data_dict_sparse, copy_data_to_new_key, apply_function_by_patient, standardize_by_patient_and_key
import math

def load_data():
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

    #log-transform sleep
    copy_data_to_new_key(dataset, "sleep_duration", "log_sleep_duration")
    apply_function_by_patient(dataset, "log_sleep_duration", lambda x: math.log(x))

    #zero-center HR
    copy_data_to_new_key(dataset, "mean_hr", "zero_centered_mean_hr")
    standardize_by_patient_and_key(dataset, "zero_centered_mean_hr", mean=0)

    #zero-center steps
    copy_data_to_new_key(dataset, "mean_steps_per_minute", "zero_centered_mean_steps_per_minute")
    standardize_by_patient_and_key(dataset, "zero_centered_mean_steps_per_minute", mean=0)
    return dataset
