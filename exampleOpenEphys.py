#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, run_cur, KilosortResults, load_qc
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si
#%%

# Change this code to load your data
data_dir = Path('/mnt/NPX/Gru/20220323/')#/2022-04-12_12-42-59/')#Path('/media/huklab/Data/NPX/Spikesorting/Combining/Gru_2022-0412_Probe1/') #Path('/home/ryanress/code/DataHorwitzLGN/data/raw/2024-12-10_Chihiro/2024-12-10_15-40-46')
stream_name = "Record Node 101#Neuropix-PXI-100.0"

subfolders=[f for f in Path(data_dir).iterdir() if f.is_dir()]

# Try to load all experiments
seg_all = si.read_openephys(subfolders[0], load_sync_timestamps=False, stream_name=stream_name, experiment_names="experiment1")# experiment_names="experiment1")

if len(subfolders) > 1:
    for subfolder in subfolders[1:]:
        print(f'Loading {subfolder}')
        seg = si.read_openephys(subfolder, load_sync_timestamps=False, stream_name=stream_name, experiment_names="experiment1")# experiment_names="experiment1")
        seg_all=si.concatenate_recordings([seg_all, seg])#seg_all.add_recording_segment(seg)

#%% Todo, add in probe data manually by seg.set_probe
import probeinterface 
record_node = "Record Node 101"
exp_id = 1
settings_file= seg.neo_reader.folder_structure[record_node]["experiments"][exp_id]["settings_file"]
if Path(settings_file).is_file():
                probe = probeinterface.read_openephys(
                    settings_file=settings_file, stream_name=stream_name, raise_error=False
                )
seg_all=seg_all.set_probe(probe, in_place=False)
#%%
# run pipelines
pipeline_dir = Path('pipeline_results')
pipeline_dir.mkdir(parents=True, exist_ok=True)

# condition signal runs 1) bad channel detection 2) 
seg_pre = condition_signal(seg_all, cache_dir=pipeline_dir / 'conditioning', recalc=False)
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False)
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')

#%% Kilosort4 parameters
# OpenEphys
sorter_params = get_default_sorter_params('kilosort4')
sorter_params['do_correction'] = False # Turns off drift correction
sorter_params['save_extra_vars'] = True # required for truncation qc
sorter_params['Th_universal'] = 10
sorter_params['Th_learned'] = 8
#sorter_params['tmin'] = 0 # doesn't seem to be supported
#sorter_params['tmax'] = 300
sorter_params['duplicate_spike_ms'] = 0.5
sorter_params['ccg_threshold'] = 0.25 #increased from 0.25, to account for long recordings where similar/same units trade off but have shared spikes
sorter_params = dict(sorter_params, **sorter_params)


#%%
try:
    ks4_results = KilosortResults(pipeline_dir / 'kilosort4')
    if (pipeline_dir / 'qc').exists():
        shutil.rmtree(pipeline_dir / 'qc')
    qc_results = load_qc(pipeline_dir / 'qc')
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


