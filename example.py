#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc, run_cur, load_cur
from spikeinterface.sorters import get_default_sorter_params
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si

#%% Change this code to load your data
local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting_true = se.read_mearec(local_path)
print(recording)
print(sorting_true)

w_ts = sw.plot_timeseries(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting_true, time_range=(0, 5))


#%% Run on a snippet to check params
start_time = 9500 - 4743 #lots of motion around 10000s in, but time didn't start at 0?
stop_time  = start_time + 1000 
seg=seg.frame_slice(start_time * 30000, stop_time * 30000) #100 seconds snippet, if really low will need to change n_batches down from 50 to 5 in condition_signal ln137

#%%
# run pipelines
pipeline_dir = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results_aftercatgt')
pipeline_dir.mkdir(parents=True, exist_ok=True)

# condition signal runs 1) bad channel detection 2) 
seg_pre = condition_signal(seg, cache_dir=pipeline_dir / 'conditioning', recalc=False)

# Motion issue on long recordings, kilosort4 is actually more robust??
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False, method='all')
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')

#%% Test data curation step
# from spikeinterface.core import load_extractor
# pipeline_dir = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results')
# seg_saved = load_extractor(pipeline_dir / 'preprocessed_recording')
# #ks4_sorter = load_extractor(pipeline_dir / 'kilosort4/sorter')
# ks4_results = KilosortResults(pipeline_dir / 'kilosort4/sorter_output') #load up the kilosort results

# shutil.rmtree(pipeline_dir / 'cur')
# cur_results = run_cur(seg_saved, ks4_sorter, pipeline_dir / 'cur', recalc=False) # this should save out some merges


#%% Kilosort4 parameters
# OpenEphys
sorter_params = get_default_sorter_params('kilosort4')
sorter_params['do_correction'] = False # Turns off drift correction
sorter_params['save_extra_vars'] = True # required for truncation qc
sorter_params['Th_universal'] = 11
sorter_params['Th_learned'] = 8
#sorter_params['tmin'] = 0 # doesn't seem to be supported
#sorter_params['tmax'] = 300
sorter_params['duplicate_spike_ms'] = 0.5
sorter_params['ccg_threshold'] = 0.35 #increased from 0.25, to account for long recordings where similar/same units trade off but have shared spikes
sorter_params = dict(sorter_params, **sorter_params)

#%%
try:
    ks4_results = KilosortResults(pipeline_dir / 'kilosort4')
    if (pipeline_dir / 'qc').exists():
        shutil.rmtree(pipeline_dir / 'qc')
    qc_results = load_qc(pipeline_dir / 'qc')
    cur_results = load_cur(pipeline_dir / 'cur')
except Exception as e:
    print(f'Failed to load sorter or qc with error:\n{e}\nRunning the pipeline again')
    seg_saved = save_binary_recording(seg_motion, pipeline_dir / 'preprocessed_recording', recalc=False)
    [ks4_results,ks4_sorter] = sort_ks4(seg_saved, pipeline_dir / 'kilosort4', sorter_params=sorter_params, recalc=False)

    qc_results = run_qc(seg_saved, ks4_results, pipeline_dir / 'qc', recalc=False)
    cur_results = run_cur(seg_saved, ks4_sorter, pipeline_dir / 'cur', recalc=False) # this should save out some merges

# Remove the processed binary
# if (pipeline_dir / 'preprocessed_recording').exists():
#     print('Removing preprocessed recording')
#     shutil.rmtree(pipeline_dir / 'preprocessed_recording')

print(f'Finished processing')

#%%


