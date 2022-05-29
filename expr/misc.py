import numpy as np
import tensorflow as tf

from util.types import Types


def beh_to_tf(all_beh):
    resp_time_left = all_beh['resp_time_left']
    resp_time_right = all_beh['resp_time_right']
    stim_hist = all_beh['stim_hist']
    trial_index = all_beh['trial_num']

    stim_hist_tf = tf.convert_to_tensor(stim_hist, Types.TF_FLOAT)
    resp_time_left_tf = tf.convert_to_tensor(resp_time_left, Types.TF_FLOAT)
    resp_time_right_tf = tf.convert_to_tensor(resp_time_right, Types.TF_FLOAT)
    trial_index_tf = tf.convert_to_tensor(trial_index, tf.int32)



    resp_time_left_tf = resp_time_left_tf[:,np.newaxis]
    resp_time_right_tf= resp_time_right_tf[:,np.newaxis]
    trial_index_tf =trial_index_tf[:,np.newaxis]

    return stim_hist_tf, resp_time_left_tf, resp_time_right_tf, trial_index_tf