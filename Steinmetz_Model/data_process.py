import pickle

import numpy as np
from util import DLogger
from util.paths import Paths


class DataProcess:
    @classmethod
    def build_inputs_trial_based(cls,
                                 spike_diffs,
                                 spike_times,
                                 beh_data,
                                 batch_size=50000):

        if batch_size is None:
            spike_index = np.arange(0, spike_times.shape[0])
        else:
            if spike_times.shape[0] >= batch_size:
                spike_index = np.random.choice(np.arange(0, spike_times.shape[0]), size=batch_size, replace=False)
            else:
                spike_index = np.arange(0, spike_times.shape[0])

        # in the paper they consider responses happening 0.1 sec before the stim
        # https: // github.com / nsteinme / steinmetz - et - al - 2019 / blob / master / behavior / findMoveTimes.m

        # beh_data[0,beh_data[6:, :].sum(axis=0) >= 1] = beh_data[0,beh_data[6:, :].sum(axis=0) >= 1] - 0.1
        # beh_data = beh_data[:, np.argsort(beh_data[0])]

        spike_diffs = spike_diffs[spike_index] + 1e-10
        spike_CHFN = spike_diffs[:, np.newaxis]
        spike_times = spike_times[spike_index ]
        beh_times = beh_data[0]

        spike_beh_hist = np.zeros((spike_diffs.shape[0], 4, beh_data.shape[0]))
        total_events = np.zeros((spike_diffs.shape[0]))
        left_hist = []
        right_hist = []
        left_CHFN = []
        right_CHFN = []

        # trial input: [feedback, go_Cue, resp, vis stim]
        trial_input = np.zeros((4, beh_data.shape[0]))

        trial_end = False
        for t in range(0, beh_times.shape[0]):

            # visual stimulus
            if beh_data[6:, t].sum() >= 1:

                if trial_end:
                    cond = np.logical_and(spike_times > trial_input[3, 0], spike_times < beh_data[0, t])

                    spike_beh_hist[cond, 3, 0] = spike_times[cond] - trial_input[3, 0]
                    spike_beh_hist[cond, 3, 1:] = trial_input[3, 1:]
                    total_events[cond] = 1

                    if trial_input[2, 0] > 0:
                        cond = np.logical_and(cond, spike_times > trial_input[2, 0])
                        spike_beh_hist[cond, 2, 0] = spike_times[cond] - trial_input[2, 0]
                        spike_beh_hist[cond, 2, 1:] = trial_input[2, 1:]
                        total_events[cond] += 1

                    cond = np.logical_and(cond, spike_times > trial_input[1, 0])
                    spike_beh_hist[cond, 1, 0] = spike_times[cond] - trial_input[1, 0]
                    spike_beh_hist[cond, 1, 1:] = trial_input[1, 1:]
                    total_events[cond] += 1

                    cond = np.logical_and(cond, spike_times > trial_input[0, 0])
                    spike_beh_hist[cond, 0, 0] = spike_times[cond] - trial_input[0, 0]
                    spike_beh_hist[cond, 0, 1:] = trial_input[0, 1:]
                    total_events[cond] += 1

                    trial_end = False
                    trial_input = np.zeros((4, beh_data.shape[0]))
                trial_input[3, 0] = beh_data[0, t]
                trial_input[3, 1:] += beh_data[1:, t]

            # resp stimulus
            if not trial_end and beh_data[2:4, t].sum() >= 1:
                # it is the first response
                if trial_input[2, 0] == 0:
                    trial_input[2, 0] = beh_data[0, t]
                    trial_input[2, 1:] += beh_data[1:, t]

            # go cue
            if not trial_end and beh_data[1, t].sum() >= 1:
                trial_input[1, 0] = beh_data[0, t]
                trial_input[1, 1:] += beh_data[1:, t]

            # feedback
            if not trial_end and beh_data[4:6, t].sum() >= 1:
                trial_input[0, 0] = beh_data[0, t]
                trial_input[0, 1:] += beh_data[1:, t]

                # for updating beh data
                resp = trial_input[2,]

                # if response is not no-go
                if resp[0] != 0:
                    beh_trial_input = np.zeros((4, beh_data.shape[0]))
                    beh_trial_input[resp[0] > trial_input[:, 0]] = \
                        trial_input[resp[0] > trial_input[:, 0]]
                    beh_trial_input[resp[0] > trial_input[:, 0], 0] = \
                        resp[0] - beh_trial_input[resp[0] > trial_input[:, 0], 0]
                    beh_trial_input[beh_trial_input == 0] = -1
                    if resp[2] == 1:
                        left_hist.append(beh_trial_input)
                        left_CHFN.append(resp[0] - trial_input[3, 0])
                    elif resp[3] == 1:
                        right_hist.append(beh_trial_input)
                        right_CHFN.append(resp[0] - trial_input[3, 0])
                    else:
                        raise Exception('invalid action')

                trial_end = True


        spike_beh_hist[:, :, 1:] = spike_beh_hist[:, :, :1] * spike_beh_hist[:, :, 1:]
        spike_beh_hist[spike_beh_hist == 0] = -1
        seq_lengths = total_events
        spike_beh_hist = spike_beh_hist[:, :, 1:]
        return seq_lengths, spike_beh_hist, spike_CHFN, \
                    np.array(left_CHFN), np.stack(left_hist, axis=0), np.array(right_CHFN), np.stack(right_hist, axis=0)


    @classmethod
    def build_inputs_trial_based_timed(cls,
                                 spike_times,
                                 spid,
                                 time_limit,
                                 beh_data,
                                 batch_size=50000):

        if spike_times is not None:
            stim_to_spike = []
            spike_stim_hist = []
            spike_id = []

            if batch_size is None:
                spike_index = np.arange(0, spike_times.shape[0])
            else:
                if spike_times.shape[0] >= batch_size:
                    spike_index = np.random.choice(np.arange(0, spike_times.shape[0]), size=batch_size, replace=False)
                else:
                    spike_index = np.arange(0, spike_times.shape[0])

            spike_times = spike_times[spike_index]
        else:
            stim_to_spike = None
            spike_stim_hist = None
            spike_id = None


        # in the paper they consider responses happening 0.1 sec before the stim
        # https: // github.com / nsteinme / steinmetz - et - al - 2019 / blob / master / behavior / findMoveTimes.m

        # beh_data[0,beh_data[6:, :].sum(axis=0) >= 1] = beh_data[0,beh_data[6:, :].sum(axis=0) >= 1] - 0.1
        # beh_data = beh_data[:, np.argsort(beh_data[0])]

        beh_times = beh_data[0]

        # trial input: [feedback, go_Cue, resp, vis stim]
        trial_input = np.zeros((4, beh_data.shape[0]))

        resp_time_left = []
        resp_time_right = []
        stim_hist = []
        trial_end_times = []
        cur_trial = 0
        spike_trial_num = []
        trial_num = []

        trial_end = False
        for t in range(0, beh_times.shape[0]):

            # visual stimulus
            if beh_data[6:, t].sum() >= 1:

                if trial_end:

                    resp_time = trial_input[2, 0]
                    stim_start = trial_input[3, 0]
                    resp = trial_input[2, ]

                    if resp_time == 0:
                        end_time = time_limit + stim_start
                    else:
                        end_time = np.min((time_limit + stim_start, resp_time))

                    trial_end_times.append(end_time - stim_start)

                    if spike_times is not None:
                        cond = np.logical_and(spike_times > stim_start, spike_times < end_time)
                        stim_to_spike.append(spike_times[cond] - stim_start)
                        spike_stim_hist.append(np.repeat(trial_input[np.newaxis, 3], cond.sum(), axis=0))
                        spike_trial_num.append(np.repeat(cur_trial, cond.sum(), axis=0))
                        spike_id.append(spid[cond])

                    stim_hist.append(trial_input[3, ])
                    trial_num.append(cur_trial)
                    if resp_time - stim_start > time_limit:
                        resp_time_left.append(-1)
                        resp_time_right.append(-1)
                    else:
                        if resp[2] == 1:
                            resp_time_left.append(resp_time - stim_start)
                            resp_time_right.append(-1)

                        elif resp[3] == 1:
                            resp_time_right.append(resp_time - stim_start)
                            resp_time_left.append(-1)
                        elif resp_time == 0:
                            resp_time_left.append(-1)
                            resp_time_right.append(-1)

                    ##########################################
                    trial_end = False
                    cur_trial += 1
                    trial_input = np.zeros((4, beh_data.shape[0]))
                trial_input[3, 0] = beh_data[0, t]
                trial_input[3, 1:] += beh_data[1:, t]

            # resp stimulus
            if not trial_end and beh_data[2:4, t].sum() >= 1:
                # it is the first response
                if trial_input[2, 0] == 0:
                    trial_input[2, 0] = beh_data[0, t]
                    trial_input[2, 1:] += beh_data[1:, t]

            # go cue
            if not trial_end and beh_data[1, t].sum() >= 1:
                trial_input[1, 0] = beh_data[0, t]
                trial_input[1, 1:] += beh_data[1:, t]

            # feedback
            if not trial_end and beh_data[4:6, t].sum() >= 1:
                trial_input[0, 0] = beh_data[0, t]
                trial_input[0, 1:] += beh_data[1:, t]

                trial_end = True

        return np.array(resp_time_left), \
               np.array(resp_time_right), \
               np.array(stim_hist), \
               np.array(trial_end_times), \
               np.array(trial_num), \
               None if spike_times  is None else np.concatenate(spike_stim_hist), \
               None if spike_times  is None else np.concatenate(spike_id), \
               None if spike_times  is None else np.concatenate(stim_to_spike), \
               None if spike_times is None else np.concatenate(spike_trial_num)

    @classmethod
    def build_inputs_history_based(cls, neural_data, beh_data, history_len, beh_data_mu, beh_data_std, batch_size=50000):
        spike_times = neural_data

        if batch_size is None:
            spike_index = np.arange(0, spike_times.shape[0] - 1)
        else:
            if spike_times.shape[0] - 1 >= batch_size:
                spike_index = np.random.choice(np.arange(0, spike_times.shape[0] - 1), size=batch_size, replace=False)
            else:
                spike_index = np.arange(0, spike_times.shape[0] - 1)

        spike_diffs = np.ediff1d(spike_times)[spike_index] + 1e-10
        input_CHFN = spike_diffs[:, np.newaxis]
        spike_times = spike_times[spike_index + 1]
        beh_times = beh_data[0]
        RNN_features = beh_data.shape[0]

        # for using fixed history
        input_RNN = np.zeros((spike_diffs.shape[0], history_len, beh_data.shape[0]))
        total_assigned = np.zeros(spike_times.shape[0])
        for t in range(beh_times.shape[0] - 1, -1, -1):
            _ = (spike_times > beh_times[t]) * (total_assigned < history_len)
            input_RNN[_, total_assigned[_].astype(int), 0] = spike_times[_] - beh_times[t]
            input_RNN[_, total_assigned[_].astype(int), 1:RNN_features] = beh_data[1:RNN_features, t]
            total_assigned[_] += 1

        input_RNN[:, :, 0] = (input_RNN[:, :, 0] - beh_data_mu) / beh_data_std

        seq_lengths = total_assigned

        return seq_lengths, input_RNN, input_CHFN

    @classmethod
    def build_model_inputs(cls, all_data, limited=False, models_42=True):

        DLogger.logger().debug("Generating inputs...")
        DLogger.logger().debug("42 regions: " + str(models_42))
        sess_data = {}
        n = len(all_data)

        if limited:
            sessions = [36, 37]
        else:
            sessions = range(n)

        for sess in sessions:
            DLogger.logger().debug("Session: " + str(sess))
            d = all_data[sess]
            beh_data = d['beh_data']
            neural_regions = sorted(d['neural_data'].keys())
            ng_data = {}
            if limited:
                ngs = sorted(list(neural_regions))[0:2]
                # ngs = ['left-VISp']
            else:
                ngs = sorted(list(neural_regions))
            for ng in ngs:
                if (not models_42) or (ng in DataProcess.get_42_regions()):
                    DLogger.logger().debug("Session: " + str(ng))
                    neural_data = d['neural_data'][ng]
                    n_neruons = neural_data['#neuron']

                    da = {}
                    seq_lengths, spike_beh_hist, spike_CHFN, \
                    left_CHFN, left_hist, right_CHFN, right_hist = \
                        DataProcess.build_inputs_trial_based(neural_data['spikes_diffs'],
                                                         neural_data['spikes_times'],
                                                         beh_data, None)
                    da['seq_lengths'] = seq_lengths
                    da['spike_beh_hist'] = spike_beh_hist
                    da['spike_CHFN'] = spike_CHFN
                    da['left_CHFN'] = left_CHFN
                    da['right_CHFN'] = right_CHFN
                    da['left_hist'] = left_hist
                    da['right_hist'] = right_hist
                    da['sp_id'] = neural_data['sp_id']
                    ng_data[ng] = da

            sess_data[sess] = ng_data
        return sess_data

    @classmethod
    def build_model_inputs_timed(cls,
                                 all_data,
                                 time_limit,
                                 sessions=None,
                                 models_42=True):

        region_index = {}
        DLogger.logger().debug("Generating inputs...")
        DLogger.logger().debug("42 regions: " + str(models_42))
        sess_data = {}
        n = len(all_data)

        if sessions == None:
            sessions = range(n)

        prev_trial_num = 0
        for sess in sessions:
            DLogger.logger().debug("Session: " + str(sess))
            d = all_data[sess]
            beh_data = d['beh_data']

            sess_data[sess] = {'beh': {}}

            resp_time_left, \
            resp_time_right, \
            stim_hist, \
            trial_end_times, \
            trial_num, \
            _, _, _, _ = \
                DataProcess.build_inputs_trial_based_timed(
                    None,
                    None,
                    time_limit,
                    beh_data,
                    None)

            da = {'resp_time_left': resp_time_left,
                  'resp_time_right': resp_time_right,
                  'stim_hist': stim_hist[:, 6:],
                  'trial_num': trial_num + prev_trial_num,
                  'trial_end_times': trial_end_times}
            sess_data[sess]['beh'] = da

            neural_regions = sorted(d['neural_data'].keys())
            ng_data = {}
            ngs = sorted(list(neural_regions))
            for ng in ngs:
                if (not models_42) or (ng in DataProcess.get_42_regions()):
                    DLogger.logger().debug("Session: " + str(ng))
                    neural_data = d['neural_data'][ng]
                    if not (ng in region_index):
                        region_index[ng] = len(region_index)

                    da = {}
                    _, \
                    _, \
                    _, \
                    _, \
                    _, \
                    spike_stim_hist, \
                    spike_id, \
                    stim_to_spike,\
                    spike_trial_num \
                        = \
                        DataProcess.build_inputs_trial_based_timed(
                                                         neural_data['spikes_times'],
                                                         neural_data['sp_id'],
                                                         time_limit,
                                                         beh_data, None)

                    da['spike_stim_hist'] = spike_stim_hist[:, 6:]
                    da['spike_id'] = spike_id
                    da['spike_trial_num'] = spike_trial_num + prev_trial_num
                    da['stim_to_spike'] = stim_to_spike
                    da['#neuron'] = neural_data['#neuron']
                    ng_data[ng] = da

            prev_trial_num += trial_num.shape[0]
            sess_data[sess]['neural'] = ng_data



        # extractign beh data from all regions
        resp_time_left_all = []
        resp_time_right_all = []
        stim_hist_all = []
        trial_num_all = []
        for sess in sorted(sess_data.keys()):
            d = sess_data[sess]
            sub_data = d['beh']

            resp_time_left_all.append(sub_data['resp_time_left'])
            resp_time_right_all.append(sub_data['resp_time_right'])
            stim_hist_all.append(sub_data['stim_hist'])
            trial_num_all.append(sub_data['trial_num'])

        region_data = {}
        for sess in sorted(sess_data.keys()):
            d = sess_data[sess]
            neural_regions = sorted(d['neural'].keys())
            for ng in sorted(list(neural_regions)):
                if not ng in region_data:
                    region_data[ng] = {
                        'spike_stim_hist': [],
                        'spike_id': [],
                        'stim_to_spike': [],
                        'trial_end_times': [],
                        'stim_hist': [],
                        'spike_trial_num': [],
                        '#neuron': [],
                        'trial_num': []
                    }
                da = region_data[ng]
                da['spike_stim_hist'].append(sess_data[sess]['neural'][ng]['spike_stim_hist'])
                da['spike_id'].append(sess_data[sess]['neural'][ng]['spike_id'])
                da['stim_to_spike'].append(sess_data[sess]['neural'][ng]['stim_to_spike'])
                da['trial_end_times'].append(sess_data[sess]['beh']['trial_end_times'])
                da['stim_hist'].append(sess_data[sess]['beh']['stim_hist'])
                da['trial_num'].append(sess_data[sess]['beh']['trial_num'])
                da['spike_trial_num'].append(sess_data[sess]['neural'][ng]['spike_trial_num'])
                da['#neuron'].append(sess_data[sess]['neural'][ng]['#neuron'] *
                                     np.ones(sess_data[sess]['beh']['stim_hist'].shape[0]))

        for ng in sorted(list(region_data.keys())):
            da = region_data[ng]
            da['spike_stim_hist'] = np.concatenate(da['spike_stim_hist'])
            da['spike_id'] = np.concatenate(da['spike_id'])
            da['stim_to_spike'] = np.concatenate(da['stim_to_spike'])
            da['trial_end_times'] = np.concatenate(da['trial_end_times'])
            da['spike_trial_num'] = np.concatenate(da['spike_trial_num'])
            da['stim_hist'] = np.concatenate(da['stim_hist'])
            da['trial_num'] = np.concatenate(da['trial_num'])
            da['#neuron'] = np.concatenate(da['#neuron'])

        return sess_data, region_index,\
                region_data, \
               {'stim_hist': np.concatenate(stim_hist_all),
                'resp_time_right': np.concatenate(resp_time_right_all),
                'resp_time_left': np.concatenate(resp_time_left_all),
                'trial_num': np.concatenate(trial_num_all),
                }, \
                prev_trial_num


    @classmethod
    def get_42_regions(cls):
        regions = ['SCs', 'VISp', 'LP', 'VISl', 'VISpm', 'VISam', 'VISrl',
         'VISa', 'LD', 'CP',
         'ACA', 'MOs', 'PL', 'GPe', 'SCm', 'MRN', 'SNr',
         'ZI', 'APN', 'ORB', 'ILA', 'ACB',
         'MOp', 'RSP', 'SSp',
         'VPL', 'LGd', 'MD', 'PO', 'POL', 'MG', 'RT', 'VPM',
         'DG', 'CA3', 'CA1', 'POST', 'SUB',
         'LS', 'PAG', 'OLF', 'BLA']

        r = ['left-' + x for x in regions]
        return r

    @staticmethod
    def load_from_file():
        with open(Paths.data_path() + 'data/processed_data/sessions/train.pickle', 'rb') as f:
            train_dict = pickle.load(f)

        region_index, region_data, all_beh, total_trial_num = \
            train_dict['region_index'], train_dict['region_data'], \
            train_dict['all_beh'], train_dict['total_trial_num'],

        with open(Paths.data_path() + 'data/processed_data/sessions/test.pickle', 'rb') as f:
            test_dict = pickle.load(f)

        test_region_data, test_all_beh, test_total_trial_num = \
            test_dict['test_region_data'], \
            test_dict['test_all_beh'], test_dict['test_total_trial_num']

        return region_index, region_data, all_beh, total_trial_num, test_region_data, test_all_beh, test_total_trial_num


if __name__ == '__main__':
    # DataProcess.get_42_regions()

    region_index, region_data, all_beh, total_trial_num, test_region_data, test_all_beh, test_total_trial_num = \
            DataProcess.load_from_file()
