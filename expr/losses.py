import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from expr.params import GetInps
from util.types import Types

@tf.function(
    input_signature=(
            # trial_end_times,
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # trial_scaling_factor,
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # spike_stim_hist,
            tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
            # stim_to_spike,
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # stim_hist,
            tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
            # spike_trial_end_times,
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # spike_scaling_factor,
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # resp_time_right_tf,
            tf.TensorSpec(shape=[None, 1], dtype=Types.TF_FLOAT),
            # resp_time_left_tf,
            tf.TensorSpec(shape=[None, 1], dtype=Types.TF_FLOAT),
            # stim_hist_tf,
            tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
            # trial_index_tf,
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            # time_limit,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # beh_norm_factor,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # resp_mean,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # resp_std,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # region_index,
            tf.TensorSpec(shape=[], dtype=tf.int32),
            # neuro_norm_factor,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # sel_trial_index,
            tf.TensorSpec(shape=[None], dtype=tf.bool),
            # sel_spike_index,
            tf.TensorSpec(shape=[None], dtype=tf.bool),
            # neuro_loss_weight,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # beh_loss_weight,
            tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # n_neuron
            tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT)),
)
def training_loop(
                trial_end_times,
                trial_scaling_factor,
                spike_stim_hist, stim_to_spike,
                stim_hist,
                spike_trial_end_times,
                spike_scaling_factor,
                resp_time_right_tf, resp_time_left_tf,
                stim_hist_tf, trial_index_tf, time_limit, beh_norm_factor, resp_mean, resp_std,
                region_index, neuro_norm_factor, sel_trial_index, sel_spike_index, neuro_loss_weight,
                beh_loss_weight, n_neuron
                  ):
    with tf.GradientTape() as grad_tape:
        loss_right, loss_left= get_beh_loses(
                                              trial_end_times,
                                              trial_scaling_factor,
                                              resp_time_right_tf,
                                              resp_time_left_tf,
                                              stim_hist_tf,
                                              trial_index_tf,
                                              time_limit,
                                              beh_norm_factor,
                                              resp_mean,
                                              resp_std)

        loss_neural = get_neural_loss(
                                      spike_stim_hist,
                                      stim_to_spike,
                                      stim_hist,
                                      trial_end_times,
                                      spike_scaling_factor,
                                      trial_scaling_factor,
                                      spike_trial_end_times,
                                      region_index,
                                      neuro_norm_factor, sel_trial_index, sel_spike_index, resp_mean,
                                      resp_std,
                                      n_neuron
                                      )
        model = GetInps.model
        optimizer = GetInps.opt
        loss_total = neuro_loss_weight * loss_neural + (loss_left + loss_right) * beh_loss_weight
        trainable_vars = model.trainable_weights
        grads = grad_tape.gradient(loss_total, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss_neural, loss_right, loss_left, loss_total


@tf.function(
    input_signature=(
            # trial_end_times,
        tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # trial_scaling_factor,
        tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
            # resp_time_right_tf,
        tf.TensorSpec(shape=[None, 1], dtype=Types.TF_FLOAT),
            # resp_time_left_tf,
        tf.TensorSpec(shape=[None, 1], dtype=Types.TF_FLOAT),
            # stim_hist_tf,
        tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
            # trial_index_tf,
        tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            # time_limit,
        tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # beh_norm_factor,
        tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # resp_mean,
        tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
            # resp_std
        tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT))
)

def get_beh_loses(
                  trial_end_times,
                  trial_scaling_factor,
                  resp_time_right_tf,
                  resp_time_left_tf,
                  stim_hist_tf,
                  trial_index_tf,
                  time_limit,
                  beh_norm_factor,
                  resp_mean,
                  resp_std):

    model = GetInps.model
    stim_hist_tf_batch = stim_hist_tf
    resp_time_tf_batch = trial_end_times
    resp_time_right_tf_batch = resp_time_right_tf
    resp_time_left_tf_batch = resp_time_left_tf
    scaling_fac_batch = trial_scaling_factor


    trial_end_times = trial_end_times[:,np.newaxis]
    trial_scaling_factor = trial_scaling_factor[:,np.newaxis]


    scaling_fac_batch = scaling_fac_batch[:,np.newaxis]
    # resp_time_right_tf_batch = tf.squeeze(resp_time_right_tf_batch)
    # resp_time_left_tf_batch = tf.squeeze(resp_time_left_tf_batch)

    with tf.GradientTape(watch_accessed_variables=False) as tape:

        w_right = tf.boolean_mask(resp_time_right_tf_batch, resp_time_right_tf_batch != -1)
        scaling_factor_right = tf.boolean_mask(scaling_fac_batch, resp_time_right_tf_batch != -1)

        tape.watch(w_right)

        # No RT

        _, Int_l_right, _, _ = model([tf.boolean_mask(stim_hist_tf_batch, resp_time_right_tf_batch[:, 0] != -1, axis=0),
                                      w_right[:, np.newaxis], scaling_factor_right[:, np.newaxis]])



        l_right = tape.gradient(Int_l_right, w_right)


        # NO RT
        _, Int_l_right, _, _ = model([stim_hist_tf_batch,
                                       0 * stim_hist_tf_batch[:, :1] + time_limit, scaling_fac_batch
                                      ])



        loss_right = -K.sum(K.log(1e-5 + l_right) / beh_norm_factor) - K.sum(-Int_l_right / beh_norm_factor)


    with tf.GradientTape(watch_accessed_variables=False) as tape:
        w_left = tf.boolean_mask(resp_time_left_tf_batch, resp_time_left_tf_batch != -1)
        scaling_factor_left = tf.boolean_mask(scaling_fac_batch, resp_time_left_tf_batch != -1)

        tape.watch(w_left)

        # NO RT
        Int_l_left, _, _, _ = model([tf.boolean_mask(stim_hist_tf_batch, resp_time_left_tf_batch[:, 0] != -1, axis=0),
                                     w_left[:, np.newaxis], scaling_factor_left[:, np.newaxis]])



        l_left = tape.gradient(Int_l_left, w_left)

        # No RT
        Int_l_left, _, _, _ = model([stim_hist_tf_batch,
                                     0 * stim_hist_tf_batch[:, :1] + time_limit, scaling_fac_batch
                                     ])



        loss_left = -K.sum(K.log(1e-5 + l_left) / beh_norm_factor) - K.sum(-Int_l_left / beh_norm_factor)
    return loss_right, loss_left


@tf.function(input_signature=(
        # spike_stim_hist,
      tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
        # stim_to_spike,
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
        # stim_hist,
      tf.TensorSpec(shape=[None, GetInps.get_stim_size()], dtype=Types.TF_FLOAT),
        # trial_end_times,
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
        # spike_scaling_factor,
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
        # trial_scaling_factor,
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
        # spike_trial_end_times,
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
        # region_index,
      tf.TensorSpec(shape=[], dtype=tf.int32),
        # neuro_norm_factor,
      tf.TensorSpec(shape=[], dtype=tf.float32),
        # sel_trial_index,
      tf.TensorSpec(shape=[None], dtype=tf.bool),
        # sel_spike_index,
      tf.TensorSpec(shape=[None], dtype=tf.bool),
        # resp_mean,
      tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
        # resp_std,
      tf.TensorSpec(shape=[], dtype=Types.TF_FLOAT),
        # n_neuron
      tf.TensorSpec(shape=[None], dtype=Types.TF_FLOAT),
      ))
def get_neural_loss(
                    spike_stim_hist,
                    stim_to_spike,
                    stim_hist,
                    trial_end_times,
                    spike_scaling_factor,
                    trial_scaling_factor,
                    spike_trial_end_times,
                    region_index,
                    neuro_norm_factor,
                    sel_trial_index,
                    sel_spike_index,
                    resp_mean,
                    resp_std,
                    n_neuron
                    ):

    model = GetInps.model



    stim_to_spike_tf = stim_to_spike[sel_spike_index]
    spike_hist_batch_tf = spike_stim_hist[sel_spike_index]
    scaling_fac_batch = spike_scaling_factor[sel_spike_index]


    trial_scaling_fac_batch = trial_scaling_factor[sel_trial_index]
    ng_hist_batch_tf = stim_hist[sel_trial_index]
    ng_trial_end_tf = trial_end_times[sel_trial_index]

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(stim_to_spike_tf)
        _, _, _, Int_l_neural = model([spike_hist_batch_tf,
                                       stim_to_spike_tf[:, np.newaxis],
                                       scaling_fac_batch[:, np.newaxis]])


        l_neural = tape.gradient(Int_l_neural[:, region_index], stim_to_spike_tf)

        _, _, _, Int_l_neural = model([ng_hist_batch_tf,
                                       ng_trial_end_tf[:, np.newaxis],
                                       trial_scaling_fac_batch[:, np.newaxis]])


        Int_l_neural = Int_l_neural[:, region_index] * n_neuron[sel_trial_index]

    loss_neural = -K.sum(K.log(1e-5 + l_neural) / neuro_norm_factor) - K.sum(-Int_l_neural / neuro_norm_factor)
    return loss_neural
