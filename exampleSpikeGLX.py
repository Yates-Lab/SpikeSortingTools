#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc, run_cur, load_cur
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si
#%%

# Change this code to load your data
data_dir = Path('/mnt/NPX/Rocky/20240704/Rocky20240704_V1V2_g0/')#/2022-04-12_12-42-59/')#Path('/media/huklab/Data/NPX/Spikesorting/Combining/Gru_2022-0412_Probe1/') #Path('/home/ryanress/code/DataHorwitzLGN/data/raw/2024-12-10_Chihiro/2024-12-10_15-40-46')
stream_id = "imec0.ap"
seg = si.read_spikeglx(folder_path=r'/mnt/NPX/Rocky/20240704/Rocky20240704_V1V2_g0/', load_sync_channel=False, stream_id=stream_id)# experiment_names="experiment1")

#%%
#seg=seg.frame_slice(0, 3000000) #100 seconds snippet, if really low will need to change n_batches down from 50 to 5 in condition_signal ln137
#seg=seg.time_slice(0, 10) #10seconds snippet, broken...
#%%
# run pipelines
pipeline_dir = Path('pipeline_results')
pipeline_dir.mkdir(parents=True, exist_ok=True)

# condition signal runs 1) bad channel detection 2) 
seg_pre = condition_signal(seg, cache_dir=pipeline_dir / 'conditioning', recalc=False)
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False)
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')
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
    [ks4_results,ks4_sorter] = sort_ks4(seg_saved, pipeline_dir / 'kilosort4', recalc=False)

    qc_results = run_qc(seg_saved, ks4_results, pipeline_dir / 'qc', recalc=False)
    cur_results = run_cur(seg_saved, ks4_sorter, pipeline_dir / 'cur', recalc=False) # this should save out some merges

# Remove the processed binary
# if (pipeline_dir / 'preprocessed_recording').exists():
#     print('Removing preprocessed recording')
#     shutil.rmtree(pipeline_dir / 'preprocessed_recording')

print(f'Finished processing')

#%%


