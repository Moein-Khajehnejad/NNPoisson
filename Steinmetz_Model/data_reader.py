import pickle
import numpy as np
import tensorflow as tf
import sys
from util.paths import Paths
from util.types import Types


def read_data(time_limit,n_regions):
####### Input data:#######
    base_dir = Paths.data_path()
    with open(base_dir+'data/steinmetz/train.pickle', 'rb') as f:
        train_dict = pickle.load(f)
        region_index, region_data, all_beh, total_trial_num = train_dict['region_index'], train_dict['region_data'], \
                                                              train_dict['all_beh'], train_dict['total_trial_num']


    beh_data = np.zeros(shape=(3,total_trial_num))
    trials_unsorted = all_beh['trial_num']
    beh_data[0,:] = all_beh['trial_num']
    beh_data[1,:] = all_beh['resp_time_left']
    beh_data[2,:] = all_beh['resp_time_right']
    sorted_beh_data = beh_data [ :, beh_data[0].argsort()]
    all_beh['trial_num'] = sorted_beh_data[0,:]
    all_beh['resp_time_left'] = sorted_beh_data[1,:]
    all_beh['resp_time_right'] = sorted_beh_data[2,:]

    stim_with_trial_indices = np.zeros(shape=(total_trial_num,9))
    stim_with_trial_indices[:,0] = trials_unsorted
    stim_with_trial_indices[:,1:9] = all_beh['stim_hist']
    sorted_stim_with_trial_indices = stim_with_trial_indices [ stim_with_trial_indices[:,0].argsort(),:]
    all_beh['stim_hist'] = sorted_stim_with_trial_indices[:,1:9]

    trial_end_time_full = {}
    k = 0
    all_trial_ends = np.zeros(shape=(n_regions, total_trial_num))
    for ng in sorted(region_data.keys()):
        temp = np.zeros(shape=(1, total_trial_num))
        indices = train_dict['region_data'][ng]['trial_num']
        temp.put(indices, train_dict['region_data'][ng]['trial_end_times'])
        trial_end_time_full[ng] = temp
        all_trial_ends[k, :] = trial_end_time_full[ng]
        k = k + 1
    all_trial_end_times = np.zeros(shape=(1, total_trial_num))
    all_trial_end_times = np.amax(all_trial_ends, axis=0)

    # %%
### Response Time Normalization Factors
    resp_mean = all_trial_end_times.mean(axis=0)
    resp_std = all_trial_end_times.std(axis=0)
    resp_mean = tf.cast(resp_mean, tf.float32)
    resp_std = tf.cast(resp_std, tf.float32)


    ### Scaling spike times:
    region_trial_index = {}
    for ng in sorted(region_data.keys()):
        indices = region_data[ng]['spike_trial_num']
        scaling = (np.repeat(time_limit, len(region_data[ng]['spike_trial_num']))) / trial_end_time_full[ng][:, indices]
        region_data[ng]['spike_scaling_factor'] = np.squeeze(np.array(scaling).T)
        region_data[ng]['spike_trial_end_times'] = (np.squeeze(trial_end_time_full[ng][:, indices]))
        region_data[ng]['trial_scaling_factor'] = np.array(time_limit / region_data[ng]['trial_end_times'])
        region_trial_index[ng] = np.array(region_data[ng]['trial_num'])

    # %%

    with open(base_dir+'/data/steinmetz/test.pickle', 'rb') as f:
        test_dict = pickle.load(f)
        test_region_data, test_all_beh, test_total_trial_num = test_dict['test_region_data'], test_dict['test_all_beh'], \
                                                               test_dict[
                                                                   'test_total_trial_num']


    test_beh_data = np.zeros(shape=(3, test_total_trial_num))
    test_trials_unsorted = test_all_beh['trial_num']
    test_beh_data[0, :] = test_all_beh['trial_num']
    test_beh_data[1, :] = test_all_beh['resp_time_left']
    test_beh_data[2, :] = test_all_beh['resp_time_right']
    sorted_test_beh_data = test_beh_data[:, test_beh_data[0].argsort()]
    test_all_beh['trial_num'] = sorted_test_beh_data[0, :]
    test_all_beh['resp_time_left'] = sorted_test_beh_data[1, :]
    test_all_beh['resp_time_right'] = sorted_test_beh_data[2, :]

    test_stim_with_trial_indices = np.zeros(shape=(test_total_trial_num,9))
    test_stim_with_trial_indices[:,0] = test_trials_unsorted
    test_stim_with_trial_indices[:,1:9] = test_all_beh['stim_hist']
    test_sorted_stim_with_trial_indices = test_stim_with_trial_indices [ test_stim_with_trial_indices[:,0].argsort(),:]
    test_all_beh['stim_hist'] = test_sorted_stim_with_trial_indices[:,1:9]

    test_trial_end_time_full = {}
    k = 0
    test_all_trial_ends = np.zeros(shape=(n_regions, test_total_trial_num))
    for ng in sorted(test_region_data.keys()):
        temp = np.zeros(shape=(1, test_total_trial_num))
        indices = test_dict['test_region_data'][ng]['trial_num']
        temp.put(indices, test_dict['test_region_data'][ng]['trial_end_times'])
        test_trial_end_time_full[ng] = temp
        test_all_trial_ends[k, :] = test_trial_end_time_full[ng]
        k = k + 1
    test_all_trial_end_times = np.zeros(shape=(1, test_total_trial_num))
    test_all_trial_end_times = np.amax(test_all_trial_ends, axis=0)

    # %%

    # Scaling spike times
    test_region_trial_index = {}
    for ng in sorted(test_region_data.keys()):
        indices = test_region_data[ng]['spike_trial_num']
        scaling = (np.repeat(time_limit, len(test_region_data[ng]['spike_trial_num'])) / test_trial_end_time_full[ng][:,
                                                                                         indices])
        test_region_data[ng]['spike_scaling_factor'] = np.squeeze(np.array(scaling).T)
        test_region_data[ng]['spike_trial_end_times'] = (np.squeeze(test_trial_end_time_full[ng][:, indices]))
        test_region_data[ng]['trial_scaling_factor'] = np.array(time_limit / test_region_data[ng]['trial_end_times'])
        test_region_trial_index[ng] = np.array(test_region_data[ng]['trial_num'])

    test_all_trial_scaling_factor = np.array(time_limit / test_all_trial_end_times )
    all_trial_scaling_factor = np.array(time_limit / all_trial_end_times )
    return region_index, region_data, all_beh, total_trial_num,region_trial_index, all_trial_end_times, all_trial_scaling_factor,\
           test_region_data, test_all_beh, test_total_trial_num,test_region_trial_index, test_all_trial_scaling_factor, \
           test_all_trial_end_times, resp_mean, resp_std



