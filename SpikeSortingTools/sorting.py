from .preprocess import get_default_job_kwargs
from spikeinterface.sorters import run_sorter, get_default_sorter_params
from pathlib import Path
import shutil
from spikeinterface.core import load_extractor
import numpy as np
import pandas as pd
from functools import cached_property

class KilosortResults:
    def __init__(self, directory):
        if isinstance(directory, str):
            directory = Path(directory)
        assert isinstance(directory, Path), 'directory must be a string or Path object'
        assert directory.exists(), f'{directory} does not exist'
        assert directory.is_dir(), f'{directory} is not a directory'
        self.directory = directory

        # Move directory to sorter_output if it is a kilosort4 output directory
        if (directory / 'sorter_output').exists():
            directory = directory / 'sorter_output'

        self.spike_times_file = directory / 'spike_times.npy'
        assert self.spike_times_file.exists(), f'{self.spike_times_file} does not exist'

        self.spike_amplitudes_file = directory / 'amplitudes.npy'
        assert self.spike_amplitudes_file.exists(), f'{self.spike_amplitudes_file} does not exist'

        self.st_file = directory / 'full_st.npy'
        if not self.st_file.exists():
            print(f'Warning: {self.st_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self.kept_spikes_file = directory / 'kept_spikes.npy'
        if not self.kept_spikes_file.exists():
            print(f'Warning: {self.kept_spikes_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')

        self.spike_clusters_file = directory / 'spike_clusters.npy'
        assert self.spike_clusters_file.exists(), f'{self.spike_clusters_file} does not exist'

        self.spike_templates_file = directory / 'spike_templates.npy'
        assert self.spike_templates_file.exists(), f'{self.spike_templates_file} does not exist'

        self.cluster_labels_file = directory / 'cluster_KSLabel.tsv'
        assert self.cluster_labels_file.exists(), f'{self.cluster_labels_file} does not exist'

        # check if ephys_metadata.json exists two levels up
        ephys_metadata_file = directory.parent.parent / 'ephys_metadata.json'
        if ephys_metadata_file.exists():
            import json
            with open(ephys_metadata_file, 'r') as f:
                self.ephys_metadata = json.load(f)

        # check if ../spikeinterface_log.json exists
        spikeinterface_log_file = directory.parent / 'spikeinterface_log.json'
        if spikeinterface_log_file.exists():
            import json
            with open(spikeinterface_log_file, 'r') as f:
                self.spikeinterface_log = json.load(f)
        
    @cached_property
    def spike_times(self):
        '''
        This now properly returns times if that info is available
        '''
        spike_times = np.load(self.spike_times_file) / 30000
        return spike_times
    
    @cached_property
    def spike_samples(self):
        return np.load(self.spike_times_file)
    
    @cached_property
    def spike_amplitudes(self):
        return self.st[:,2]

    @cached_property
    def st(self): 
        st = np.load(self.st_file)
        spikes = np.load(self.kept_spikes_file)
        return st[spikes]
    
    @cached_property
    def spike_clusters(self):
        return np.load(self.spike_clusters_file)

    @cached_property
    def spike_templates(self):
        return np.load(self.spike_templates_file)

    @cached_property
    def cluster_labels(self):
        return pd.read_csv(self.cluster_labels_file, sep='\t')


def save_binary_recording(seg, cache_dir, recalc=False, job_kwargs={}):
    '''
    Save a given spikeinterface extractor to a binary format. If the cache_dir exists,
    then will attempt to load from there. If the extractor cannot be loaded, then the extractor is saved.
    Saving a preprocessed recording reduces computation time when running the sorter, especially if
    running multiple sorters.

    Parameters:
    ------------
    seg: spikeinterface extractor
        The extractor to save
    cache_dir: str or Path
    recalc: bool
        If True, will delete the cache_dir and rerun the sorter

    Returns:
    -----------
    seg_saved: spikeinterface extractor
        The loaded output
    '''
    if recalc:
        shutil.rmtree(cache_dir)

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if cache_dir.exists():
        try:
            seg_load = load_extractor(cache_dir)
        except Exception as e:
            print(f'Failed to load extractor: {e}')
            shutil.rmtree(cache_dir)
    
    if not cache_dir.exists():
        _job_kwargs = get_default_job_kwargs()
        _job_kwargs.update(job_kwargs)
        seg.save(folder=cache_dir, **_job_kwargs)

    return load_extractor(cache_dir)

def sort_ks4(seg, cache_dir, sorter_params = {}, recalc=False):
    '''
    Sort a given spikeinterface extractor using kilosort4. If the cache_dir exists,
    then will attempt to loaded from there. If the sorting cannot be loaded, then kilsort4 is run.

    Parameters:
    ------------
    seg: spikeinterface extractor
        The extractor to sort
    cache_dir: str or Path
    sorter_params: dict
        Parameters to pass to the sorter
    recalc: bool
        If True, will delete the cache_dir and rerun the sorter

    Returns:
    -----------
    ks4_sorting: spikeinterface sorting extractor
        The sorted output
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if recalc and cache_dir.exists():
        shutil.rmtree(cache_dir)

    ks4_sorting = None
    if cache_dir.exists():
        try:
            ks4_sorting = KilosortResults(cache_dir / 'sorter_output')
        except Exception as e:
            print(f'Failed to load kilosort4 sorting: {e}')
            shutil.rmtree(cache_dir)

    if not cache_dir.exists():
        # Run kilosort4 locally
        params = get_default_sorter_params('kilosort4')
        params['do_correction'] = False # Turns off drift correction
        params['save_extra_vars'] = True # required for truncation qc
        params = dict(params, **sorter_params) # overwrite any default params present in sorter_params

        _ = run_sorter("kilosort4", seg, folder=str(cache_dir), verbose=True, remove_existing_folder=True, **params)
        ks4_sorting = KilosortResults(cache_dir / 'sorter_output')

    return ks4_sorting



