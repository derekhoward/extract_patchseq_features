from pathlib import Path

from ipfx.feature_extractor import (SpikeFeatureExtractor, SpikeTrainFeatureExtractor)
import ipfx.stimulus_protocol_analysis as spa
from ipfx.epochs import get_stim_epoch
from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
from ipfx.error import FeatureError
from ipfx.qc_feature_extractor import sweep_qc_features, cell_qc_features

def process_dataset_sweeps(data_set):
    drop_failed_sweeps(data_set)
    long_square_table = data_set.filtered_sweep_table(stimuli=data_set.ontology.long_square_names) 
    long_square_table = long_square_table[long_square_table['passed'] == True]
    long_square_table = long_square_table[long_square_table['clamp_mode'] == "CurrentClamp"]
    
    good_sweeps = list()
    for i in long_square_table.sweep_number:
        try:
            curr_sweep = data_set.sweep_set(i).sweeps[0]
            good_sweeps.append(i)
        except:
            print("Rejected " + str(i))

    long_square_table = long_square_table[long_square_table['sweep_number'].isin(good_sweeps)]
    long_square_sweeps = data_set.sweep_set(long_square_table.sweep_number)
    
    return long_square_sweeps


def extract_features(long_square_sweeps):
    # Select epoch corresponding to the actual recording from the sweeps
    # and align sweeps so that the experiment would start at the same time
    long_square_sweeps.select_epoch("recording")
    long_square_sweeps.align_to_start_of_epoch("experiment")   
    
    # find the start and end time of the stimulus
    # (treating the first sweep as representative)
    stim_start_index, stim_end_index = get_stim_epoch(long_square_sweeps.i[0])
    stim_start_time = long_square_sweeps.t[0][stim_start_index]
    stim_end_time = long_square_sweeps.t[0][stim_end_index]
    print(f'Start: {stim_start_time}, end: {stim_end_time}')
        
    spfx = SpikeFeatureExtractor(start=stim_start_time, end=stim_end_time, filter = 1)
    sptfx = SpikeTrainFeatureExtractor(start=stim_start_time, end=stim_end_time, baseline_interval = 0.05)

    # run the analysis and print out a few of the features
    lsa = spa.LongSquareAnalysis(spfx, sptfx, subthresh_min_amp=-100.0) #or should the subthresh min amp be -500?
    lsa_results = lsa.analyze(long_square_sweeps)
    
    return lsa_results


def generated_formatted_features_output(nwb_path):
    filename = Path(nwb_path).name
    experiment_features = {}
    experiment_features['filename'] = filename

    try:
        data_set = create_ephys_data_set(nwb_file=nwb_path)
        sweep_features = sweep_qc_features(data_set)
        qc_features, cell_tags = cell_qc_features(data_set)
        lsa_sweeps = process_dataset_sweeps(data_set)
        lsa_features = extract_features(lsa_sweeps)

        # Extract general features
        experiment_features['v_baseline'] = lsa_features['v_baseline']
        experiment_features['rheobase_i'] = lsa_features['rheobase_i']
        experiment_features['fi_fit_slope'] = lsa_features['fi_fit_slope']
        experiment_features['sag'] = lsa_features['sag']
        experiment_features['vm_for_sag'] = lsa_features['vm_for_sag']
        experiment_features['input_resistance'] = lsa_features['input_resistance']
        experiment_features['tau'] = lsa_features['tau']

        # Extract features from hero sweep
        experiment_features['hero_adapt'] = lsa_features['hero_sweep'].adapt
        experiment_features['hero_avg_rate'] = lsa_features['hero_sweep'].avg_rate
        experiment_features['hero_first_isi'] = lsa_features['hero_sweep'].first_isi
        experiment_features['hero_isi_cv'] = lsa_features['hero_sweep'].isi_cv
        experiment_features['hero_latency'] = lsa_features['hero_sweep'].latency
        experiment_features['hero_mean_isi'] = lsa_features['hero_sweep'].mean_isi
        experiment_features['hero_median_isi'] = lsa_features['hero_sweep'].median_isi
        experiment_features['hero_stim_amp'] = lsa_features['hero_sweep'].stim_amp

        # Extract features from rheobase

        # identify rheobase index sweep to extract additinal rheobase features
        rheobase_index = lsa_features['rheobase_sweep'].name
        rheobase_features = lsa_features['spikes_set'][rheobase_index]

        experiment_features['rheo_threshold_v'] = rheobase_features.loc[0, 'threshold_v']
        experiment_features['rheo_trough_v'] = rheobase_features.loc[0, 'trough_v']
        experiment_features['rheo_fast_trough_v'] = rheobase_features.loc[0, 'fast_trough_v']
        experiment_features['rheo_slow_trough_v'] = rheobase_features.loc[0, 'slow_trough_v']
        experiment_features['rheo_adp_v'] = rheobase_features.loc[0, 'adp_v']
        experiment_features['rheo_width'] = rheobase_features.loc[0, 'width']
        experiment_features['rheo_upstroke_downstroke_ratio'] = rheobase_features.loc[0, 'upstroke_downstroke_ratio']
        experiment_features['rheo_peak_t'] = rheobase_features.loc[0, 'peak_t']
        experiment_features['rheo_fast_trough_t'] = rheobase_features.loc[0, 'fast_trough_t']
        experiment_features['rheo_trough_t'] = rheobase_features.loc[0, 'trough_t']
        experiment_features['rheo_slow_trough_t'] = rheobase_features.loc[0, 'slow_trough_t']
        experiment_features['rheo_peak_v'] = lsa_features['rheobase_sweep'].peak_deflect[0]

        # identify maximal firing index sweep to extract additinal maximal firing features
        experiment_features['maximal_firing_rate'] = lsa_features['sweeps']['avg_rate'].values.max()

        # identify qc_features
        experiment_features['qc_blowout_mv'] = qc_features['blowout_mv']
        experiment_features['qc_electrode_0_pa'] = qc_features['electrode_0_pa']
        experiment_features['qc_recording_date'] = qc_features['recording_date']
        experiment_features['qc_seal_gohm'] = qc_features['seal_gohm']
        experiment_features['qc_input_resistance_mohm'] = qc_features['input_resistance_mohm']
        experiment_features['qc_initial_access_resistance_mohm'] = qc_features['initial_access_resistance_mohm']
        experiment_features['qc_input_access_resistance_ratio'] = qc_features['input_access_resistance_ratio']
        print(cell_tags)
    
    except (FeatureError, ValueError, TypeError, KeyError) as e:
        print(f'Error in {nwb_path}: {e}')
    
    return experiment_features