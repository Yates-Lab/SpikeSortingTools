#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si
#%%

# Change this code to load your data
data_dir = Path('/home/ryanress/code/DataHorwitzLGN/data/raw/2024-12-10_Chihiro/2024-12-10_15-40-46')
stream_name = "Record Node 101#Neuropix-PXI-100.ProbeA-AP"
seg = si.read_openephys(data_dir, stream_name=stream_name)

#%%
# run pipeline
pipeline_dir = Path('pipeline_results')
pipeline_dir.mkdir(parents=True, exist_ok=True)

# condition signal runs 1) bad channel detection 2) 
seg_pre = condition_signal(seg, cache_dir=pipeline_dir / 'conditioning', recalc=False)
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False)
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')

try:
    ks4_results = KilosortResults(pipeline_dir / 'kilosort4')
    if (pipeline_dir / 'qc').exists():
        shutil.rmtree(pipeline_dir / 'qc')
    qc_results = load_qc(pipeline_dir / 'qc')
except Exception as e:
    print(f'Failed to load sorter or qc with error:\n{e}\nRunning the pipeline again')
    seg_saved = save_binary_recording(seg_motion, pipeline_dir / 'preprocessed_recording', recalc=False)
    ks4_results = sort_ks4(seg_saved, pipeline_dir / 'kilosort4', recalc=False)
    qc_results = run_qc(seg_saved, ks4_results, pipeline_dir / 'qc', recalc=False)

# Remove the processed binary
if (pipeline_dir / 'preprocessed_recording').exists():
    print('Removing preprocessed recording')
    shutil.rmtree(pipeline_dir / 'preprocessed_recording')

print(f'Finished processing')

#%%


