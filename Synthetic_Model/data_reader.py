import pickle
from util.paths import Paths
import numpy as np
import tensorflow as tf
import pandas as pd

def read_data(time_limit):
    ####### Input data:#######
    base_dir = Paths.output_path()
    with open( base_dir +'/data/synth/train_sample.pickle', 'rb') as f:

        train_dict = pd.read_pickle(f)
        region_index, region_data, all_beh, total_trial_num = train_dict['region_index'], train_dict['region_data'],\
                                                              train_dict['all_beh'], train_dict['total_trial_num']
    original_end_times = region_data['DE1s']['trial_end_times']

    ### Correcting bad trials
    bad_trials = region_data['DE1s']['trial_end_times']>2
    trials_to_correct = [i for i, x in enumerate(bad_trials) if x]

    for ng in sorted(region_data.keys()):

        spike_to_remove = np.where(region_data[ng]['stim_to_spike']>2)
        region_data[ng]['stim_to_spike'] = region_data[ng]['stim_to_spike'][region_data[ng]['stim_to_spike']<=2]
        region_data[ng]['trial_end_times'][trials_to_correct] =2
        region_data[ng]['spike_stim_hist'] = np.delete(region_data[ng]['spike_stim_hist'], (spike_to_remove), axis=0)
        region_data[ng]['spike_trial_num'][spike_to_remove] = -1
        region_data[ng]['spike_trial_num'] = region_data[ng]['spike_trial_num'][region_data[ng]['spike_trial_num']!=-1]

    all_beh['resp_time_right'][all_beh['resp_time_right'] > 2] = -1
    all_beh['resp_time_left'][all_beh['resp_time_left'] > 2] = -1

    ### Response Time Normalization Factors
    resp_mean = region_data['DE1s']['trial_end_times'].mean(axis=0) #0
    resp_std = region_data['DE1s']['trial_end_times'].std(axis=0) #1
    resp_mean = tf.cast(resp_mean, tf.float32)
    resp_std = tf.cast(resp_std, tf.float32)

    ### Scaling spike times:
    for ng in sorted(region_data.keys()):
        indices = region_data[ng]['spike_trial_num']
        scaling = (np.repeat(time_limit, len(region_data[ng]['spike_trial_num']))) /original_end_times[indices]
        region_data[ng]['spike_scaling_factor'] = np.array(scaling)
        region_data[ng]['spike_trial_end_times'] = region_data[ng]['trial_end_times'][indices]
        region_data[ng]['trial_scaling_factor'] = np.array((time_limit / np.array(region_data['DE1s']['trial_end_times'])))

    with open(base_dir +'/data/synth/test_sample.pickle', 'rb') as f:

        test_dict = pd.read_pickle(f)
        test_region_data, test_all_beh, test_total_trial_num = test_dict['region_data'], test_dict['all_beh'], test_dict[
            'total_trial_num']

    #### Removing bad trials
    test_original_end_times = test_region_data['DE1s']['trial_end_times']

    ### Correcting bad trials
    bad_trials = test_region_data['DE1s']['trial_end_times']>2
    trials_to_correct = [i for i, x in enumerate(bad_trials) if x]

    for ng in sorted(test_region_data.keys()):

        spike_to_remove = np.where(test_region_data[ng]['stim_to_spike']>2)
        test_region_data[ng]['stim_to_spike'] = test_region_data[ng]['stim_to_spike'][test_region_data[ng]['stim_to_spike']<=2]
        test_region_data[ng]['trial_end_times'][trials_to_correct] = 2
        test_region_data[ng]['spike_stim_hist'] = np.delete(test_region_data[ng]['spike_stim_hist'], (spike_to_remove), axis=0)
        test_region_data[ng]['spike_trial_num'][spike_to_remove] = -1
        test_region_data[ng]['spike_trial_num'] = test_region_data[ng]['spike_trial_num'][test_region_data[ng]['spike_trial_num']!=-1]

    test_all_beh['resp_time_right'][test_all_beh['resp_time_right'] > 2] = -1
    test_all_beh['resp_time_left'][test_all_beh['resp_time_left'] > 2] = -1

    for ng in sorted(test_region_data.keys()):
        indices = test_region_data[ng]['spike_trial_num']
        scaling = (np.repeat(time_limit, len(test_region_data[ng]['spike_trial_num'])) /test_original_end_times[indices])
        test_region_data[ng]['spike_scaling_factor'] = np.array(scaling)
        test_region_data[ng]['spike_trial_end_times'] = test_original_end_times[indices]
        test_region_data[ng]['trial_scaling_factor'] = np.array(
            (time_limit / np.array(test_region_data['DE1s']['trial_end_times'])))

    return region_index, region_data, all_beh, total_trial_num, \
           test_region_data, test_all_beh, test_total_trial_num, resp_mean, resp_std



