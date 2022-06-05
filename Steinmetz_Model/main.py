import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sys import exit
from expr.params import GetInps
from Steinmetz_Model.data_reader import read_data

# ##  (No RT):
stim_shape = (8,)
GetInps.stim_size = stim_shape[0]
from expr.losses import get_beh_loses, get_neural_loss, training_loop


from expr.misc import beh_to_tf
from expr.model import build_model
from util import DLogger
from util.helper import ensure_dir, fix_seeds
from util.logger import LogFile
from util.paths import Paths
from util.types import Types

sys.path.append("../../")  # go to parent dir
n_regions = 42
iterations = 2000
episodes = 10
beh_norm_factor = tf.constant(200, Types.TF_FLOAT)
neuro_norm_factor = tf.constant(200 * n_regions, Types.TF_FLOAT)
time_limit = tf.constant(0.4, dtype=Types.TF_FLOAT)  # for Steinmetz data
n_batches = int(600 / 60)

def get_configs(n_regions):
    configs = []
    for size_stim_layer1 in [10]:
        for size_stim_layer2 in [10]:
            for size_stim_layer3 in [10]:
                for size_time_layer in [10]:
                    for size_StimTime_layer in [1]:
                        for size_neural_layer1 in [10]:
                            for size_neural_layer2 in [n_regions]:
                                for size_beh_layer1 in [10]:
                                    for size_beh_layer2 in [1]:
                                        for lr in [0.0001]:
                                            #                                                 for dropout in [1]:
                                            for flt in ['32']:
                                                configs.append({
                                                    'size_stim_layer1': size_stim_layer1,
                                                    'size_stim_layer2': size_stim_layer2,
                                                    'size_stim_layer3': size_stim_layer3,
                                                    'size_time_layer': size_time_layer,
                                                    'size_StimTime_layer': size_StimTime_layer,
                                                    'size_neural_layer1': size_neural_layer1,
                                                    'size_neural_layer2': size_neural_layer2,
                                                    'size_beh_layer1': size_beh_layer1,
                                                    'size_beh_layer2': size_beh_layer2,
                                                    'lr': lr,
                                                    'flt': flt,
                                                    'n_regions': n_regions

                                                })
    return configs


if __name__ == '__main__':

    configs = get_configs(n_regions)
    fix_seeds()

    region_index, region_data, all_beh, total_trial_num, region_trial_index, all_trial_end_times, all_trial_scaling_factor, \
    test_region_data, test_all_beh, test_total_trial_num, test_region_trial_index , \
    test_all_trial_scaling_factor, test_all_trial_end_times, resp_mean, resp_std = read_data(time_limit,n_regions)
    test_all_trial_scaling_factor_tf = tf.convert_to_tensor(test_all_trial_scaling_factor, Types.TF_FLOAT)
    test_all_trial_end_times_tf = tf.convert_to_tensor(test_all_trial_end_times, Types.TF_FLOAT)

    for conf_index in range(len(configs)):
        config = configs[conf_index]
        flt = config['flt']
        if flt == '32':
            Types.TF_FLOAT = tf.float32
            Types.TF_FLOAT_str = 'float32'
        elif flt == '64':
            Types.TF_FLOAT = tf.float64
            Types.TF_FLOAT_str = 'float64'
        else:
            raise Exception('invalid float point')

        tf.keras.backend.set_floatx(Types.TF_FLOAT_str)
        model = build_model(conf_index, stim_shape, config)
        lr = config['lr']
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        sel_trial_index_cached = {}
        sel_spike_index_cached = {}
        for ng in sorted(region_data.keys()):
            neural_data = region_data[ng]
            spike_trial_index = neural_data['spike_trial_num']
            ng_trial_index = tf.squeeze(neural_data['trial_num'])
            total_trials = np.random.permutation(np.unique(ng_trial_index))
            trial_bins = np.array_split(total_trials, n_batches)
            sel_trial_index_cached[ng] = {}
            sel_spike_index_cached[ng] = {}
            for b in range(n_batches):
                cindex = trial_bins[b]
                sel_trial_index_cached[ng][b] = tf.convert_to_tensor(np.isin(ng_trial_index, cindex))
                sel_spike_index_cached[ng][b] = tf.convert_to_tensor(np.isin(spike_trial_index, cindex))

        test_sel_trial_index_cached = {}
        test_sel_spike_index_cached = {}
        for ng in sorted(test_region_data.keys()):
            neural_data = test_region_data[ng]
            spike_trial_index = neural_data['spike_trial_num']
            ng_trial_index = tf.squeeze(neural_data['trial_num'])
            ung_trial = np.unique(ng_trial_index)
            cindex = ung_trial
            test_sel_trial_index_cached[ng] = tf.convert_to_tensor(np.isin(ng_trial_index, cindex))
            test_sel_spike_index_cached[ng] = tf.convert_to_tensor(np.isin(spike_trial_index, cindex))

        dir_name = ''
        for k in sorted(config.keys()):
            dir_name += str(k) + '_' + str(config[k]) + '_'

        base_dir = Paths.output_path()
        output = base_dir + 'results/local/tanh/NeuroPixel/TestForDebugging_' + dir_name + '/'

        GetInps.model = model
        GetInps.opt = optimizer

        with LogFile(output, 'run.log'):

            DLogger.logger().debug(output)

            model.summary(print_fn=lambda x: DLogger.logger().debug(x))

            pd.DataFrame.from_dict(region_index, orient='index').to_csv(output + 'region_index.csv')

            stim_hist_tf, resp_time_left_tf, resp_time_right_tf, trial_index_tf = beh_to_tf(all_beh)
            test_stim_hist_tf, test_resp_time_left_tf, test_resp_time_right_tf, test_trial_index_tf = beh_to_tf(
                test_all_beh)

            for ng in region_data.keys():
                region_index[ng] = tf.constant(region_index[ng])

                for neural_data in [region_data[ng]]:
                    neural_data['spike_stim_hist'] = tf.convert_to_tensor(neural_data['spike_stim_hist'],
                                                                          Types.TF_FLOAT)
                    neural_data['stim_to_spike'] = tf.convert_to_tensor(neural_data['stim_to_spike'], Types.TF_FLOAT)
                    neural_data['#neuron'] = tf.convert_to_tensor(neural_data['#neuron'], Types.TF_FLOAT)
                    neural_data['stim_hist'] = tf.convert_to_tensor(neural_data['stim_hist'], Types.TF_FLOAT)
                    neural_data['trial_end_times'] = tf.convert_to_tensor(neural_data['trial_end_times'],
                                                                          Types.TF_FLOAT)
                    neural_data['spike_trial_num'] = tf.convert_to_tensor(neural_data['spike_trial_num'],
                                                                          Types.TF_FLOAT)
                    neural_data['trial_num'] = tf.convert_to_tensor(neural_data['trial_num'], Types.TF_FLOAT),
                    neural_data['spike_scaling_factor'] = tf.convert_to_tensor(neural_data['spike_scaling_factor'],
                                                                               Types.TF_FLOAT)
                    neural_data['trial_scaling_factor'] = tf.convert_to_tensor(neural_data['trial_scaling_factor'],
                                                                               Types.TF_FLOAT)
                    neural_data['spike_trial_end_times'] = tf.convert_to_tensor(neural_data['spike_trial_end_times'],
                                                                                Types.TF_FLOAT)
                    neural_data['#neuron'] = tf.convert_to_tensor(neural_data['#neuron'], Types.TF_FLOAT)

            for ng in test_region_data.keys():
                region_index[ng] = tf.constant(region_index[ng])
                for neural_data in [test_region_data[ng]]:
                    neural_data['spike_stim_hist'] = tf.convert_to_tensor(neural_data['spike_stim_hist'],
                                                                          Types.TF_FLOAT)
                    neural_data['stim_to_spike'] = tf.convert_to_tensor(neural_data['stim_to_spike'],
                                                                        Types.TF_FLOAT)
                    neural_data['#neuron'] = tf.convert_to_tensor(neural_data['#neuron'], Types.TF_FLOAT)
                    neural_data['stim_hist'] = tf.convert_to_tensor(neural_data['stim_hist'], Types.TF_FLOAT)
                    neural_data['trial_end_times'] = tf.convert_to_tensor(neural_data['trial_end_times'],
                                                                          Types.TF_FLOAT)
                    neural_data['spike_trial_num'] = tf.convert_to_tensor(neural_data['spike_trial_num'],
                                                                          Types.TF_FLOAT)
                    neural_data['trial_num'] = tf.convert_to_tensor(neural_data['trial_num'], Types.TF_FLOAT),
                    neural_data['spike_scaling_factor'] = tf.convert_to_tensor(neural_data['spike_scaling_factor'],
                                                                               Types.TF_FLOAT)
                    neural_data['trial_scaling_factor'] = tf.convert_to_tensor(neural_data['trial_scaling_factor'],
                                                                               Types.TF_FLOAT)
                    neural_data['spike_trial_end_times'] = tf.convert_to_tensor(
                        neural_data['spike_trial_end_times'],
                        Types.TF_FLOAT)
                    neural_data['#neuron'] = tf.convert_to_tensor(neural_data['#neuron'], Types.TF_FLOAT)




            all_test = []
            test_response_times_tf = {}
            response_times_tf = {}

            for it in range(iterations):
                beh_loss_weight = tf.constant(0.01, Types.TF_FLOAT)
                neuro_loss_weight = tf.constant(1, Types.TF_FLOAT)

                if (it % 20) == 0:
                    _ = output + "models-" + str(it) + '/'
                    ensure_dir(_)
                    model.save(_ + 'model-fwd.h5', include_optimizer=False)

                epoch_loss_neural = 0
                epoch_loss_total = 0
                epoch_loss_left = 0
                epoch_loss_right = 0
                for e in range(episodes):
                    for b in range(int(n_batches)):
                        for ng in sorted(region_data.keys()):
                            neural_data = region_data[ng]
                            sel_trial_index = sel_trial_index_cached[ng][b]
                            sel_spike_index = sel_spike_index_cached[ng][b]
                            loss_neural, loss_right, loss_left, loss_total = training_loop(
                                neural_data['trial_end_times'], neural_data['trial_scaling_factor'],
                                neural_data['spike_stim_hist'], neural_data['stim_to_spike'],
                                neural_data['stim_hist'],
                                neural_data['spike_trial_end_times'],
                                neural_data['spike_scaling_factor'],
                                tf.convert_to_tensor(all_beh['resp_time_right'][(np.array(neural_data['trial_num'])).astype(int)].T, dtype=Types.TF_FLOAT),
                                tf.convert_to_tensor(all_beh['resp_time_left'][(np.array(neural_data['trial_num'])).astype(int)].T, dtype=Types.TF_FLOAT),
                                tf.convert_to_tensor(np.squeeze(all_beh['stim_hist'][(np.array(neural_data['trial_num'])).astype(int)]), dtype=Types.TF_FLOAT),
                                trial_index_tf, time_limit, beh_norm_factor, resp_mean, resp_std,
                                region_index[ng], neuro_norm_factor, sel_trial_index, sel_spike_index, neuro_loss_weight,
                                beh_loss_weight, neural_data['#neuron']
                            )
                            epoch_loss_neural += loss_neural
                            epoch_loss_left += loss_left
                            epoch_loss_right += loss_right
                            epoch_loss_total += loss_total

                test_eval = {}

                test_eval['train loss neural'] = epoch_loss_neural.numpy()
                test_eval['train loss total'] = epoch_loss_total.numpy()
                test_eval['train loss left'] = epoch_loss_left.numpy()
                test_eval['train loss right'] = epoch_loss_right.numpy()

                test_loss_right, test_loss_left = get_beh_loses(
                    test_all_trial_end_times_tf,
                    test_all_trial_scaling_factor_tf,
                    test_resp_time_right_tf,
                    test_resp_time_left_tf,
                    test_stim_hist_tf,
                    test_trial_index_tf,
                    time_limit,
                    beh_norm_factor,
                    resp_mean,
                    resp_std)

                test_eval['iter'] = it
                test_eval['test loss left'] = test_loss_left.numpy()
                test_eval['test loss right'] = test_loss_right.numpy()

                test_neural_loss_total = 0
                for ng in sorted(test_region_data.keys()):
                    neural_data = test_region_data[ng]
                    sel_trial_index = test_sel_trial_index_cached[ng]
                    sel_spike_index = test_sel_spike_index_cached[ng]
                    test_loss_neural = get_neural_loss(
                        test_region_data[ng]['spike_stim_hist'],
                        test_region_data[ng]['stim_to_spike'],
                        test_region_data[ng]['stim_hist'],
                        test_region_data[ng]['trial_end_times'],
                        test_region_data[ng]['spike_scaling_factor'],
                        test_region_data[ng]['trial_scaling_factor'],
                        test_region_data[ng]['spike_trial_end_times'],
                        region_index[ng],
                        neuro_norm_factor, sel_trial_index, sel_spike_index, resp_mean,
                        resp_std, test_region_data[ng]['#neuron'])

                    test_eval[ng] = test_loss_neural.numpy()
                    test_neural_loss_total += test_loss_neural

                test_loss_total = test_neural_loss_total * neuro_loss_weight + (
                        test_loss_left + test_loss_right) * beh_loss_weight
                test_eval['test loss total'] = test_loss_total.numpy()
                test_eval['test loss neural'] = test_neural_loss_total.numpy()

                all_test.append(test_eval)
                pd.DataFrame(all_test).to_csv(output + '/test.csv')

                # test_eval['train loss total'] = float('NaN')
                DLogger.logger().debug(f"iter = {it:04d} "
                                       f"train total loss = {test_eval['train loss total']:7.3f} "
                                       f"train loss neural = {test_eval['train loss neural']:7.3f} "
                                       f"train loss left = {test_eval['train loss left']:7.3f} "
                                       f"train loss right = {test_eval['train loss right']:7.3f} "
                                       f"test loss total = {test_eval['test loss total']:7.3f} "
                                       f"test loss neural = {test_eval['test loss neural']:7.3f} "
                                       f"test loss left = {test_eval['test loss left']:7.3f} "
                                       f"test loss right = {test_eval['test loss right']:7.3f} "
                                       )
                if (np.isnan(test_eval['train loss total']) | np.isnan(test_eval['test loss total'])):
                    exit()


