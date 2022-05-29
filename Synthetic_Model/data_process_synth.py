import pickle
import numpy as np

from util.paths import Paths


def alt(a, end, window, start=0, step=1):
    bin_starts = np.arange(start, end+1-window, step)
    bin_ends = bin_starts + window
    last_index = np.searchsorted(a, bin_ends, side='right')
    first_index = np.searchsorted(a, bin_starts, side='left')
    return  last_index - first_index

def sliding_count(a, end, window, start=0, step=1):
    bins = [(x, x + window) for x in range(start, (end + 1) - window, step)]
    counts = np.zeros(len(bins))
    for i, rng in enumerate(bins):
        count = len(a[np.where(np.logical_and(a>=rng[0], a<=rng[1]))])
        counts[i] = count
    return counts


class DataProcessSynth:
    @classmethod
    def build_inputs(cls, file_dirs):
        thr = 40
        n_sims = 600
        choices = {}
        cohs = np.load(file_dirs + "cohs.npy", allow_pickle=True, encoding='bytes')
        stim1 = np.load(file_dirs + "stim1s.npy", allow_pickle=True, encoding='bytes')
        stim2 = np.load(file_dirs + "stim2s.npy", allow_pickle=True, encoding='bytes')

        #calculating actions
        spikes1 = np.load(file_dirs + 'S_DE1s.npy', allow_pickle=True, encoding='bytes')
        spikes2 = np.load(file_dirs + 'S_DE2s.npy', allow_pickle=True, encoding='bytes')

        resp_time_right = []
        resp_time_left = []
        trial_num = []
        trial_end_times = []
        stim_hist = []
        choice_hist = []
        for sim in range(n_sims):
            choices[sim] = {}
            spikes_times1 = np.hstack(spikes1.item()[sim].values())
            spikes_times2 = np.hstack(spikes2.item()[sim].values())

            n_neuro = len(spikes1.item()[sim].values())
            #calculating sliding window count
            rate1 = sliding_count(1000 * spikes_times1, 3000, 50, step=1) / n_neuro / (50 / 1000)
            rate2 = sliding_count(1000 * spikes_times2, 3000, 50, step=1) / n_neuro / (50 / 1000)

            rate1 = rate1[500:]
            rate2 = rate2[500:]

            # +50 is to get end of the sliding window as time
            ind1 = np.where(rate1 > thr)[0]
            ind2 = np.where(rate2 > thr)[0]

            if ind1.shape[0] > 0 :
                if ind2.shape[0] > 0:
                    if ind2[0] > ind1[0]:
                        choices[sim]['R1'] = ind1[0] / 1000 + 0.05# ms to s
                        choices[sim]['R2'] = -1 # no response
                    else:
                        choices[sim]['R1'] = -1
                        choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
                else:
                    choices[sim]['R1'] = ind1[0] / 1000 + 0.05 # ms to s
                    choices[sim]['R2'] = -1  # no response
            elif ind2.shape[0] > 0:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
            else:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = -1

            stim = np.zeros([3])
            if cohs.item()[sim] == 0.8:
                stim[0] = 1
            elif cohs.item()[sim] == 0.5:
                stim[1] = 1
            elif cohs.item()[sim] == 0.0:
                stim[2] = 1
            else:
                raise Exception("unknown stimulus")

            stim_hist.append(stim)
            trial_num.append(sim)
            if choices[sim]['R1'] != -1:
                trial_end_times.append(choices[sim]['R1'])
                resp_time_right.append(choices[sim]['R1'])
                resp_time_left.append(-1)
            elif choices[sim]['R2'] != -1:
                trial_end_times.append(choices[sim]['R2'])
                resp_time_left.append(choices[sim]['R2'])
                resp_time_right.append(-1)
            else:
                #no choice was made
                trial_end_times.append(2.5)
                resp_time_left.append(-1)
                resp_time_right.append(-1)

            spike_choice = np.zeros([4])
            if resp_time_left[sim] == -1:
                spike_choice[1] = 1
            else:
                spike_choice[0] = resp_time_left[sim]
            if resp_time_right[sim] == -1:
                spike_choice[3] = 1
            else:
                spike_choice[2] = resp_time_right[sim]
            choice_hist.append(spike_choice)

        region_data = {}
        beh_data = {}
        for region in ['DE1s', 'DE2s', 'SE1s', 'SE2s']:
            spikes = np.load(file_dirs + 'S_' + region + ".npy", allow_pickle=True, encoding='bytes')
            # simulation number
            spike_stim_hist = []
            stim_to_spike = []
            spike_trial_num = []
            n_neurons = []
            spike_choice_hist = []
            for sim in range(n_sims):
                spikes_times = np.hstack(spikes.item()[sim].values())
                spikes_times = spikes_times[spikes_times > 0.5] - 0.5
                prior_resp = spikes_times < trial_end_times[sim]
                spikes_times = spikes_times[prior_resp]
                stim_to_spike.append(spikes_times)
                stim = np.zeros([spikes_times.shape[0], 3])
                stim[:, :] = stim_hist[sim]
                spike_stim_hist.append(stim)

                spike_choice = np.zeros([spikes_times.shape[0], 4])
                if resp_time_left[sim] == -1:
                    spike_choice[:, 1] = 1
                else:
                    spike_choice[:, 0] = resp_time_left[sim]
                if resp_time_right[sim] == -1:
                    spike_choice[:, 3] = 1
                else:
                    spike_choice[:, 2] = resp_time_right[sim]
                spike_choice_hist.append(spike_choice)

                n_neuro = len(spikes1.item()[sim].values())
                n_neurons.append(n_neuro)
                spike_trial_num.append(np.full(spikes_times.shape[0], sim))

            region_data[region] = {}

            # for debuggins
            #
            # np.hstack(np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values())
            #
            # ((np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) > 0.5) * \
            # (np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) < 0.923)).sum()
            #

            region_data[region]['stim_to_spike'] = np.hstack(stim_to_spike)
            region_data[region]['spike_stim_hist'] = np.vstack(spike_stim_hist)
            region_data[region]['spike_choice_hist'] = np.vstack(spike_choice_hist)
            region_data[region]['trial_end_times'] = np.hstack(trial_end_times)
            region_data[region]['#neuron'] = np.hstack(n_neurons)
            region_data[region]['stim_hist'] = np.vstack(stim_hist)
            region_data[region]['trial_num'] = np.hstack(trial_num)
            region_data[region]['spike_trial_num'] = spike_trial_num
            region_data[region]['spike_trial_num'] = np.hstack(spike_trial_num)
            region_data[region]['choice_hist'] = np.vstack(choice_hist)

        beh_data['stim_hist'] = np.vstack(stim_hist)
        beh_data['resp_time_right'] = np.vstack(resp_time_right)[:, 0]
        beh_data['resp_time_left'] = np.vstack(resp_time_left)[:, 0]
        beh_data['trial_num'] = np.hstack(trial_num)

        region_index = {'DE1s':0,
                        'DE2s':1,
                        'SE1s':2,
                        'SE2s':3}

        return region_index, region_data, beh_data, n_sims

    @classmethod
    def build_inputs_var(cls, file_dirs):
        thr = 40
        n_sims = 600
        choices = {}
        cohs = np.load(file_dirs + "cohs.npy", allow_pickle=True, encoding='bytes')
        stim1 = np.load(file_dirs + "stim1s.npy", allow_pickle=True, encoding='bytes')
        stim2 = np.load(file_dirs + "stim2s.npy", allow_pickle=True, encoding='bytes')

        #calculating actions
        spikes1 = np.load(file_dirs + 'S_DE1s.npy', allow_pickle=True, encoding='bytes')
        spikes2 = np.load(file_dirs + 'S_DE2s.npy', allow_pickle=True, encoding='bytes')

        resp_time_right = []
        resp_time_left = []
        trial_num = []
        trial_end_times = []
        stim_hist = []
        choice_hist = []
        for sim in range(n_sims):
            thr = 20 + np.random.uniform() * 30
            choices[sim] = {}
            spikes_times1 = np.hstack(spikes1.item()[sim].values())
            spikes_times2 = np.hstack(spikes2.item()[sim].values())

            n_neuro = len(spikes1.item()[sim].values())
            #calculating sliding window count
            rate1 = sliding_count(1000 * spikes_times1, 3000, 50, step=1) / n_neuro / (50 / 1000)
            rate2 = sliding_count(1000 * spikes_times2, 3000, 50, step=1) / n_neuro / (50 / 1000)

            rate1 = rate1[500:]
            rate2 = rate2[500:]

            # +50 is to get end of the sliding window as time
            ind1 = np.where(rate1 > thr)[0]
            ind2 = np.where(rate2 > thr)[0]

            if ind1.shape[0] > 0 :
                if ind2.shape[0] > 0:
                    if ind2[0] > ind1[0]:
                        choices[sim]['R1'] = ind1[0] / 1000 + 0.05# ms to s
                        choices[sim]['R2'] = -1 # no response
                    else:
                        choices[sim]['R1'] = -1
                        choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
                else:
                    choices[sim]['R1'] = ind1[0] / 1000 + 0.05 # ms to s
                    choices[sim]['R2'] = -1  # no response
            elif ind2.shape[0] > 0:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
            else:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = -1

            stim = np.zeros([3])
            if cohs.item()[sim] == 0.8:
                stim[0] = 1
            elif cohs.item()[sim] == 0.5:
                stim[1] = 1
            elif cohs.item()[sim] == 0.0:
                stim[2] = 1
            else:
                raise Exception("unknown stimulus")

            stim_hist.append(stim)
            trial_num.append(sim)
            if choices[sim]['R1'] != -1:
                trial_end_times.append(choices[sim]['R1'])
                resp_time_right.append(choices[sim]['R1'])
                resp_time_left.append(-1)
            elif choices[sim]['R2'] != -1:
                trial_end_times.append(choices[sim]['R2'])
                resp_time_left.append(choices[sim]['R2'])
                resp_time_right.append(-1)
            else:
                #no choice was made
                trial_end_times.append(2.5)
                resp_time_left.append(-1)
                resp_time_right.append(-1)

            spike_choice = np.zeros([4])
            if resp_time_left[sim] == -1:
                spike_choice[1] = 1
            else:
                spike_choice[0] = resp_time_left[sim]
            if resp_time_right[sim] == -1:
                spike_choice[3] = 1
            else:
                spike_choice[2] = resp_time_right[sim]
            choice_hist.append(spike_choice)

        region_data = {}
        beh_data = {}
        for region in ['DE1s', 'DE2s', 'SE1s', 'SE2s']:
            spikes = np.load(file_dirs + 'S_' + region + ".npy", allow_pickle=True, encoding='bytes')
            # simulation number
            spike_stim_hist = []
            stim_to_spike = []
            spike_trial_num = []
            n_neurons = []
            spike_choice_hist = []
            for sim in range(n_sims):
                spikes_times = np.hstack(spikes.item()[sim].values())
                spikes_times = spikes_times[spikes_times > 0.5] - 0.5
                prior_resp = spikes_times < trial_end_times[sim]
                spikes_times = spikes_times[prior_resp]
                stim_to_spike.append(spikes_times)
                stim = np.zeros([spikes_times.shape[0], 3])
                stim[:, :] = stim_hist[sim]
                spike_stim_hist.append(stim)

                spike_choice = np.zeros([spikes_times.shape[0], 4])
                if resp_time_left[sim] == -1:
                    spike_choice[:, 1] = 1
                else:
                    spike_choice[:, 0] = resp_time_left[sim]
                if resp_time_right[sim] == -1:
                    spike_choice[:, 3] = 1
                else:
                    spike_choice[:, 2] = resp_time_right[sim]
                spike_choice_hist.append(spike_choice)

                n_neuro = len(spikes1.item()[sim].values())
                n_neurons.append(n_neuro)
                spike_trial_num.append(np.full(spikes_times.shape[0], sim))

            region_data[region] = {}

            # for debuggins
            #
            # np.hstack(np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values())
            #
            # ((np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) > 0.5) * \
            # (np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) < 0.923)).sum()
            #

            region_data[region]['stim_to_spike'] = np.hstack(stim_to_spike)
            region_data[region]['spike_stim_hist'] = np.vstack(spike_stim_hist)
            region_data[region]['spike_choice_hist'] = np.vstack(spike_choice_hist)
            region_data[region]['trial_end_times'] = np.hstack(trial_end_times)
            region_data[region]['#neuron'] = np.hstack(n_neurons)
            region_data[region]['stim_hist'] = np.vstack(stim_hist)
            region_data[region]['trial_num'] = np.hstack(trial_num)
            region_data[region]['spike_trial_num'] = spike_trial_num
            region_data[region]['spike_trial_num'] = np.hstack(spike_trial_num)
            region_data[region]['choice_hist'] = np.vstack(choice_hist)

        beh_data['stim_hist'] = np.vstack(stim_hist)
        beh_data['resp_time_right'] = np.vstack(resp_time_right)[:, 0]
        beh_data['resp_time_left'] = np.vstack(resp_time_left)[:, 0]
        beh_data['trial_num'] = np.hstack(trial_num)

        region_index = {'DE1s':0,
                        'DE2s':1,
                        'SE1s':2,
                        'SE2s':3}

        return region_index, region_data, beh_data, n_sims

    @classmethod
    def build_inputs_var2(cls, file_dirs):
        thr = 40
        #TODO: change this to 600 for test
        n_sims = 1200
        choices = {}
        cohs = np.load(file_dirs + "cohs.npy", allow_pickle=True, encoding='bytes')
        stim1 = np.load(file_dirs + "stim1s.npy", allow_pickle=True, encoding='bytes')
        stim2 = np.load(file_dirs + "stim2s.npy", allow_pickle=True, encoding='bytes')

        #calculating actions
        spikes1 = np.load(file_dirs + 'S_DE1s.npy', allow_pickle=True, encoding='bytes')
        spikes2 = np.load(file_dirs + 'S_DE2s.npy', allow_pickle=True, encoding='bytes')

        resp_time_right = []
        resp_time_left = []
        trial_num = []
        trial_end_times = []
        stim_hist = []
        choice_hist = []
        for sim in range(n_sims):
            thr = 10 + np.random.uniform() * 50
            choices[sim] = {}
            spikes_times1 = np.hstack(spikes1.item()[sim].values())
            spikes_times2 = np.hstack(spikes2.item()[sim].values())

            n_neuro = len(spikes1.item()[sim].values())
            #calculating sliding window count
            rate1 = sliding_count(1000 * spikes_times1, 3000, 50, step=1) / n_neuro / (50 / 1000)
            rate2 = sliding_count(1000 * spikes_times2, 3000, 50, step=1) / n_neuro / (50 / 1000)

            rate1 = rate1[500:]
            rate2 = rate2[500:]

            # +50 is to get end of the sliding window as time
            ind1 = np.where(rate1 > thr)[0]
            ind2 = np.where(rate2 > thr)[0]

            if ind1.shape[0] > 0 :
                if ind2.shape[0] > 0:
                    if ind2[0] > ind1[0]:
                        choices[sim]['R1'] = ind1[0] / 1000 + 0.05# ms to s
                        choices[sim]['R2'] = -1 # no response
                    else:
                        choices[sim]['R1'] = -1
                        choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
                else:
                    choices[sim]['R1'] = ind1[0] / 1000 + 0.05 # ms to s
                    choices[sim]['R2'] = -1  # no response
            elif ind2.shape[0] > 0:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
            else:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = -1

            stim = np.zeros([3])
            if cohs.item()[sim] == 0.8:
                stim[0] = 1
            elif cohs.item()[sim] == 0.5:
                stim[1] = 1
            elif cohs.item()[sim] == 0.0:
                stim[2] = 1
            else:
                raise Exception("unknown stimulus")

            stim_hist.append(stim)
            trial_num.append(sim)
            if choices[sim]['R1'] != -1:
                trial_end_times.append(choices[sim]['R1'])
                resp_time_right.append(choices[sim]['R1'])
                resp_time_left.append(-1)
            elif choices[sim]['R2'] != -1:
                trial_end_times.append(choices[sim]['R2'])
                resp_time_left.append(choices[sim]['R2'])
                resp_time_right.append(-1)
            else:
                #no choice was made
                trial_end_times.append(2.5)
                resp_time_left.append(-1)
                resp_time_right.append(-1)

            spike_choice = np.zeros([4])
            if resp_time_left[sim] == -1:
                spike_choice[1] = 1
            else:
                spike_choice[0] = resp_time_left[sim]
            if resp_time_right[sim] == -1:
                spike_choice[3] = 1
            else:
                spike_choice[2] = resp_time_right[sim]
            choice_hist.append(spike_choice)

        region_data = {}
        beh_data = {}
        for region in ['DE1s', 'DE2s', 'SE1s', 'SE2s']:
            spikes = np.load(file_dirs + 'S_' + region + ".npy", allow_pickle=True, encoding='bytes')
            # simulation number
            spike_stim_hist = []
            stim_to_spike = []
            spike_trial_num = []
            n_neurons = []
            spike_choice_hist = []
            for sim in range(n_sims):
                spikes_times = np.hstack(spikes.item()[sim].values())
                spikes_times = spikes_times[spikes_times > 0.5] - 0.5
                prior_resp = spikes_times < trial_end_times[sim]
                spikes_times = spikes_times[prior_resp]
                stim_to_spike.append(spikes_times)
                stim = np.zeros([spikes_times.shape[0], 3])
                stim[:, :] = stim_hist[sim]
                spike_stim_hist.append(stim)

                spike_choice = np.zeros([spikes_times.shape[0], 4])
                if resp_time_left[sim] == -1:
                    spike_choice[:, 1] = 1
                else:
                    spike_choice[:, 0] = resp_time_left[sim]
                if resp_time_right[sim] == -1:
                    spike_choice[:, 3] = 1
                else:
                    spike_choice[:, 2] = resp_time_right[sim]
                spike_choice_hist.append(spike_choice)

                n_neuro = len(spikes1.item()[sim].values())
                n_neurons.append(n_neuro)
                spike_trial_num.append(np.full(spikes_times.shape[0], sim))

            region_data[region] = {}

            # for debuggins
            #
            # np.hstack(np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values())
            #
            # ((np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) > 0.5) * \
            # (np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) < 0.923)).sum()
            #

            region_data[region]['stim_to_spike'] = np.hstack(stim_to_spike)
            region_data[region]['spike_stim_hist'] = np.vstack(spike_stim_hist)
            region_data[region]['spike_choice_hist'] = np.vstack(spike_choice_hist)
            region_data[region]['trial_end_times'] = np.hstack(trial_end_times)
            region_data[region]['#neuron'] = np.hstack(n_neurons)
            region_data[region]['stim_hist'] = np.vstack(stim_hist)
            region_data[region]['trial_num'] = np.hstack(trial_num)
            region_data[region]['spike_trial_num'] = spike_trial_num
            region_data[region]['spike_trial_num'] = np.hstack(spike_trial_num)
            region_data[region]['choice_hist'] = np.vstack(choice_hist)

        beh_data['stim_hist'] = np.vstack(stim_hist)
        beh_data['resp_time_right'] = np.vstack(resp_time_right)[:, 0]
        beh_data['resp_time_left'] = np.vstack(resp_time_left)[:, 0]
        beh_data['trial_num'] = np.hstack(trial_num)

        region_index = {'DE1s':0,
                        'DE2s':1,
                        'SE1s':2,
                        'SE2s':3}

        return region_index, region_data, beh_data, n_sims


    @classmethod
    def build_inputs_sample_neuron(cls, file_dirs, n_sims):
        thr = 40
        choices = {}
        cohs = np.load(file_dirs + "cohs.npy", allow_pickle=True, encoding='bytes')
        stim1 = np.load(file_dirs + "stim1s.npy", allow_pickle=True, encoding='bytes')
        stim2 = np.load(file_dirs + "stim2s.npy", allow_pickle=True, encoding='bytes')

        #calculating actions
        spikes1 = np.load(file_dirs + 'S_DE1s.npy', allow_pickle=True, encoding='bytes')
        spikes2 = np.load(file_dirs + 'S_DE2s.npy', allow_pickle=True, encoding='bytes')

        resp_time_right = []
        resp_time_left = []
        trial_num = []
        trial_end_times = []
        stim_hist = []
        choice_hist = []
        for sim in range(n_sims):
            choices[sim] = {}
            s1 = spikes1.item()[sim]
            s2 = spikes2.item()[sim]
            idx1 = np.random.choice(list(s1.keys()), int(len(s1) / 3), False)
            idx2 = np.random.choice(list(s2.keys()), int(len(s1) / 3), False)
            spikes_times1 = []
            spikes_times2 = []
            for i1, i2 in zip(idx1, idx2):
                spikes_times1.append(s1[i1])
                spikes_times2.append(s2[i2])
            spikes_times1 = np.hstack(spikes_times1)
            spikes_times2 = np.hstack(spikes_times2)

            n_neuro1 = len(idx1)
            n_neuro2 = len(idx2)
            #calculating sliding window count
            rate1 = sliding_count(1000 * spikes_times1, 3000, 50, step=1) / n_neuro1 / (50 / 1000)
            rate2 = sliding_count(1000 * spikes_times2, 3000, 50, step=1) / n_neuro2 / (50 / 1000)

            rate1 = rate1[500:]
            rate2 = rate2[500:]

            # +50 is to get end of the sliding window as time
            ind1 = np.where(rate1 > thr)[0]
            ind2 = np.where(rate2 > thr)[0]

            if ind1.shape[0] > 0 :
                if ind2.shape[0] > 0:
                    if ind2[0] > ind1[0]:
                        choices[sim]['R1'] = ind1[0] / 1000 + 0.05# ms to s
                        choices[sim]['R2'] = -1 # no response
                    else:
                        choices[sim]['R1'] = -1
                        choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
                else:
                    choices[sim]['R1'] = ind1[0] / 1000 + 0.05 # ms to s
                    choices[sim]['R2'] = -1  # no response
            elif ind2.shape[0] > 0:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = ind2[0] / 1000 + 0.05 # ms to s
            else:
                choices[sim]['R1'] = -1
                choices[sim]['R2'] = -1

            stim = np.zeros([3])
            if cohs.item()[sim] == 0.8:
                stim[0] = 1
            elif cohs.item()[sim] == 0.5:
                stim[1] = 1
            elif cohs.item()[sim] == 0.0:
                stim[2] = 1
            else:
                raise Exception("unknown stimulus")

            stim_hist.append(stim)
            trial_num.append(sim)
            if choices[sim]['R1'] != -1:
                trial_end_times.append(choices[sim]['R1'])
                resp_time_right.append(choices[sim]['R1'])
                resp_time_left.append(-1)
            elif choices[sim]['R2'] != -1:
                trial_end_times.append(choices[sim]['R2'])
                resp_time_left.append(choices[sim]['R2'])
                resp_time_right.append(-1)
            else:
                #no choice was made
                trial_end_times.append(2.5)
                resp_time_left.append(-1)
                resp_time_right.append(-1)

            spike_choice = np.zeros([4])
            if resp_time_left[sim] == -1:
                spike_choice[1] = 1
            else:
                spike_choice[0] = resp_time_left[sim]
            if resp_time_right[sim] == -1:
                spike_choice[3] = 1
            else:
                spike_choice[2] = resp_time_right[sim]
            choice_hist.append(spike_choice)

        region_data = {}
        beh_data = {}
        for region in ['DE1s', 'DE2s', 'SE1s', 'SE2s']:
            spikes = np.load(file_dirs + 'S_' + region + ".npy", allow_pickle=True, encoding='bytes')
            # simulation number
            spike_stim_hist = []
            stim_to_spike = []
            spike_trial_num = []
            n_neurons = []
            spike_choice_hist = []
            for sim in range(n_sims):
                spikes_times = np.hstack(spikes.item()[sim].values())
                spikes_times = spikes_times[spikes_times > 0.5] - 0.5
                prior_resp = spikes_times < trial_end_times[sim]
                spikes_times = spikes_times[prior_resp]
                stim_to_spike.append(spikes_times)
                stim = np.zeros([spikes_times.shape[0], 3])
                stim[:, :] = stim_hist[sim]
                spike_stim_hist.append(stim)

                spike_choice = np.zeros([spikes_times.shape[0], 4])
                if resp_time_left[sim] == -1:
                    spike_choice[:, 1] = 1
                else:
                    spike_choice[:, 0] = resp_time_left[sim]
                if resp_time_right[sim] == -1:
                    spike_choice[:, 3] = 1
                else:
                    spike_choice[:, 2] = resp_time_right[sim]
                spike_choice_hist.append(spike_choice)

                n_neuro = len(spikes1.item()[sim].values())
                n_neurons.append(n_neuro)
                spike_trial_num.append(np.full(spikes_times.shape[0], sim))

            region_data[region] = {}

            # for debuggins
            #
            # np.hstack(np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values())
            #
            # ((np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) > 0.5) * \
            # (np.hstack(
            # np.load(file_dirs + 'S_' + "DE1s" + ".npy", allow_pickle=True, encoding='bytes').item()[8].values()) < 0.923)).sum()
            #

            region_data[region]['stim_to_spike'] = np.hstack(stim_to_spike)
            region_data[region]['spike_stim_hist'] = np.vstack(spike_stim_hist)
            region_data[region]['spike_choice_hist'] = np.vstack(spike_choice_hist)
            region_data[region]['trial_end_times'] = np.hstack(trial_end_times)
            region_data[region]['#neuron'] = np.hstack(n_neurons)
            region_data[region]['stim_hist'] = np.vstack(stim_hist)
            region_data[region]['trial_num'] = np.hstack(trial_num)
            region_data[region]['spike_trial_num'] = spike_trial_num
            region_data[region]['spike_trial_num'] = np.hstack(spike_trial_num)
            region_data[region]['choice_hist'] = np.vstack(choice_hist)

        beh_data['stim_hist'] = np.vstack(stim_hist)
        beh_data['resp_time_right'] = np.vstack(resp_time_right)[:, 0]
        beh_data['resp_time_left'] = np.vstack(resp_time_left)[:, 0]
        beh_data['trial_num'] = np.hstack(trial_num)

        region_index = {'DE1s':0,
                        'DE2s':1,
                        'SE1s':2,
                        'SE2s':3}

        return region_index, region_data, beh_data, n_sims


    @staticmethod
    def save_to_file_var():
        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_var("../nongit/data/synth/train/")
        with open('../nongit/data/synth/train_var.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_var("../nongit/data/synth/test/")
        with open('../nongit/data/synth/test_var.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)

    @staticmethod
    def save_to_file_var2():
        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_var2("../nongit/data/synth/400/")
        with open('../nongit/data/synth/train_var_2.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_var2("../nongit/data/synth/test/")
        with open('../nongit/data/synth/test_var_2.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


    @staticmethod
    def save_to_file_sample():
        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_sample_neuron("../nongit/data/synth/400/", 1200)
        with open('../nongit/data/synth/train_sample.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs_sample_neuron("../nongit/data/synth/test/", 600)
        with open('../nongit/data/synth/test_sample.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


    @staticmethod
    def save_to_file():
        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs("../nongit/data/synth/train/")
        with open('../nongit/data/synth/train.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)


        region_index, region_data, all_beh, total_trial_num = DataProcessSynth.build_inputs("../nongit/data/synth/test/")
        with open('../nongit/data/synth/test.pickle', 'wb') as f:
            pickle.dump({
                'region_index' : region_index ,
                'region_data' : region_data,
                'all_beh' : all_beh,
                'total_trial_num' : total_trial_num
            }, f)

    @staticmethod
    def load_from_file():
        with open(Paths.data_path() + '/data/synth/train.pickle', 'rb') as f:
            train_dict = pickle.load(f)

        region_index, region_data, all_beh, total_trial_num = \
            train_dict['region_index'], train_dict['region_data'], \
            train_dict['all_beh'], train_dict['total_trial_num'],

        with open(Paths.data_path() + '/data/synth/test.pickle', 'rb') as f:
            train_dict = pickle.load(f)

        _, test_region_data, test_all_beh, test_total_trial_num = \
            train_dict['region_index'], train_dict['region_data'], \
            train_dict['all_beh'], train_dict['total_trial_num'],

        return region_index, region_data, all_beh, total_trial_num, test_region_data, test_all_beh, test_total_trial_num


    @staticmethod
    def load_from_file_nothr():
        with open('../nongit/data/synth/no-thr/train.pickle', 'rb') as f:
            train_dict = pickle.load(f)

        region_index, region_data, all_beh, total_trial_num = \
            train_dict['region_index'], train_dict['region_data'], \
            train_dict['all_beh'], train_dict['total_trial_num'],

        with open('../nongit/data/synth/no-thr/test.pickle', 'rb') as f:
            train_dict = pickle.load(f)

        _, test_region_data, test_all_beh, test_total_trial_num = \
            train_dict['region_index'], train_dict['region_data'], \
            train_dict['all_beh'], train_dict['total_trial_num'],

        return region_index, region_data, all_beh, total_trial_num, test_region_data, test_all_beh, test_total_trial_num

    @staticmethod
    def plot_graphs():
        region_index, region_data, all_beh, total_trial_num, test_region_data, test_all_beh, test_total_trial_num \
            = DataProcessSynth.load_from_file()

        import numpy as np
        import matplotlib.mlab as mlab
        import matplotlib.pyplot as plt

        x = all_beh['resp_time_right'][all_beh['resp_time_right'] != -1]
        n, bins, patches = plt.hist(x, 40, facecolor='blue', alpha=0.5)
        x = all_beh['resp_time_left'][all_beh['resp_time_left'] != -1]
        n, bins, patches = plt.hist(x, 40, facecolor='red', alpha=0.5)
        plt.show()
        pass


if __name__ == '__main__':
    # DataProcessSynth.build_inputs_sample_neuron("../nongit/data/synth/train/")
    # DataProcessSynth.save_to_file_sample()
    # DataProcessSynth.load_from_file()
    DataProcessSynth.plot_graphs()