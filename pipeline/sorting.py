
from spikeinterface.extractors import read_kilosort
from pathlib import Path
import shutil
from spikeinterface.core import load_extractor
import numpy as np
from spikeinterface.extractors import read_kilosort
import pandas as pd
from spikeinterface.sorters import run_sorter

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
        self._spike_times = None

        self.spike_amplitudes_file = directory / 'amplitudes.npy'
        assert self.spike_amplitudes_file.exists(), f'{self.spike_amplitudes_file} does not exist'
        self._spike_amplitudes = None

        self.st_file = directory / 'full_st.npy'
        if not self.st_file.exists():
            print(f'Warning: {self.st_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self.kept_spikes_file = directory / 'kept_spikes.npy'
        if not self.kept_spikes_file.exists():
            print(f'Warning: {self.kept_spikes_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self._st = None

        self.spike_clusters_file = directory / 'spike_clusters.npy'
        assert self.spike_clusters_file.exists(), f'{self.spike_clusters_file} does not exist'
        self._spike_clusters = None

        self.spike_templates_file = directory / 'spike_templates.npy'
        assert self.spike_templates_file.exists(), f'{self.spike_templates_file} does not exist'
        self._spike_templates = None

        self.cluster_labels_file = directory / 'cluster_KSLabel.tsv'
        assert self.cluster_labels_file.exists(), f'{self.cluster_labels_file} does not exist'
        self._cluster_labels = None
        
    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = np.load(self.spike_times_file)
        return self._spike_times
    
    @property
    def spike_amplitudes(self):
        if self._spike_amplitudes is None:
            self._spike_amplitudes = np.load(self.spike_amplitudes_file)
        return self._spike_amplitudes

    @property
    def st(self): 
        if self._st is None:
            st = np.load(self.st_file)
            spikes = np.load(self.kept_spikes_file)
            self._st = st[spikes]
        return self._st
    
    @property
    def spike_clusters(self):
        if self._spike_clusters is None:
            self._spike_clusters = np.load(self.spike_clusters_file)
        return self._spike_clusters

    @property
    def spike_templates(self):
        if self._spike_templates is None:
            self._spike_templates = np.load(self.spike_templates_file)
        return self._spike_templates

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            self._cluster_labels = pd.read_csv(self.cluster_labels_file, sep='\t')
        return self._cluster_labels


def save_binary_recording(seg, cache_dir, recalc=False):
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
        seg.save(folder=cache_dir, n_jobs=-1)

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
    ks4_sorter = None
    if cache_dir.exists():
        try:
            ks4_sorting = KilosortResults(cache_dir / 'sorter_output')
            ks4_sorter = load_extractor(cache_dir / 'sorter')
        except Exception as e:
            print(f'Failed to load kilosort4 sorting: {e}')
            shutil.rmtree(cache_dir)

    if not cache_dir.exists():


        ks4_sorter = run_sorter("kilosort4", seg, folder=str(cache_dir), verbose=True, remove_existing_folder=True, **sorter_params)
        ks4_sorter.save_to_folder(folder=cache_dir / 'sorter')
        ks4_sorting = KilosortResults(cache_dir / 'sorter_output') # Pull from output directory

    return ks4_sorting, ks4_sorter



