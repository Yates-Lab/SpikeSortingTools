#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc, run_cur, load_cur
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si

#%% Change this code to load your data
data_dir=   r"/mnt/NPX/Rocky/20240704/Rocky20240704_V1V2_g0/"

stream_id = "imec1.ap" #usually imec0 is first inserted probe (often V2/MT), imec1 is second probe (often V1)
seg = si.read_spikeglx(folder_path=data_dir, load_sync_channel=False, stream_id=stream_id)# experiment_names="experiment1")

#%% Run on a snippet to check params
# start_time = 0 #lots of motion around 10000s in, but time didn't start at 0?
# stop_time  = start_time + 1000 
# seg=seg.frame_slice(start_time * 30000, stop_time * 30000) #100 seconds snippet, if really low will need to change n_batches down from 50 to 5 in condition_signal ln137

#%%
# run pipelines
pipeline_dir = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results_Rocky20240704_V1V2_g0_imec1')
pipeline_dir.mkdir(parents=True, exist_ok=True)

#%%
# condition signal runs 1) bad channel detection 2) 
noise_thresh = 0.2 # higher for spikeGLX, 
seg_pre = condition_signal(seg, cache_dir=pipeline_dir / 'conditioning', noise_thresh=noise_thresh, recalc=False)

#%% DEBUG: quick saving out of the preprocessed recording before motion correction
#seg_saved = save_binary_recording(seg_pre, pipeline_dir / 'preprocessed_recording', recalc=False)
#%%

# Motion issue on SpikeGLX, this may have had more to do with the conditioning failing, kilosort4 is actually more robust??
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False, method='med')
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
sorter_params['duplicate_spike_ms'] = 0.5
sorter_params['ccg_threshold'] = 0.4 #increased from 0.25, to account for long recordings where similar/same units trade off but have shared spikes
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


