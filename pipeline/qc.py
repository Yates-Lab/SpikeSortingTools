#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .refractory import compute_rvl_tensor, plot_rvl
from .truncation import analyze_amplitude_truncation, plot_amplitude_truncation
from pathlib import Path
from tqdm import tqdm


def truncation_qc(spike_times, spike_clusters, spike_amplitudes, cache_dir, recalc=False):
    '''
    Run the truncation quality control pipeline on the given sorted data.

    Parameters
    ----------
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    truncation_path = cache_dir / 'truncation_qc.npz'
    present_path = cache_dir / 'present_qc.npz'
    pdf_path = cache_dir / 'truncation_qc.pdf'

    if truncation_path.exists() and present_path.exists() and pdf_path.exists() and not recalc:
        trunc_qc = np.load(truncation_path)
        pres_qc = np.load(present_path)
        return trunc_qc, pres_qc

    cids = np.unique(spike_clusters)

    pdf = PdfPages(pdf_path)

    trunc_qc = {
        'cid': [],
        'window_blocks': [],
        'popts': [],
        'mpcts': []
    }

    pres_qc = {
        'cid': [],
        'valid_blocks': []
    }

    for cid in tqdm(cids):
        cluster_spikes = spike_times[spike_clusters == cid]
        cluster_amps = spike_amplitudes[spike_clusters == cid]
        window_blocks, valid_blocks, popts, mpcts = analyze_amplitude_truncation(cluster_spikes, cluster_amps)

        if len(window_blocks) > 0:
            trunc_qc['cid'].append(np.ones(len(window_blocks)) * cid)

            if window_blocks.ndim == 1:
                window_blocks = window_blocks[np.newaxis, :]
            trunc_qc['window_blocks'].append(window_blocks)
            
            popts = np.array(popts)
            if popts.ndim == 1:
                popts = popts[np.newaxis, :]
            trunc_qc['popts'].append(popts)

            trunc_qc['mpcts'].append(mpcts)

            pres_qc['cid'].append(np.ones(len(valid_blocks)) * cid)
            pres_qc['valid_blocks'].append(valid_blocks)
            
        fig, axs = plot_amplitude_truncation(cluster_spikes, cluster_amps, window_blocks, valid_blocks, mpcts)
        axs[0].set_title(f'Cluster {cid}\nAmplitudes vs Time')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

    trunc_qc = {k: np.concatenate(v, axis=0) for k, v in trunc_qc.items()}
    pres_qc = {k: np.concatenate(v, axis=0) for k, v in pres_qc.items()}

    np.savez(truncation_path, **trunc_qc)
    np.savez(present_path, **pres_qc)

    return trunc_qc, pres_qc

def refractory_qc(spike_times, spike_clusters, cache_dir, recalc=False):
    '''
    Run the refractory period quality control pipeline on the given sorted data.
    
    Parameters
    ----------
    spike_times: array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters: array-like (n_spikes,)
        The cluster assignments of each spike.
    
    Returns
    -------
    qc_results: dict
        The results of the quality control pipeline
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'refractory_qc.npz'
    pdf_path = cache_dir / 'refractory_qc.pdf'
    if npz_path.exists() and pdf_path.exists() and not recalc:
        qc_results = np.load(npz_path)
        return qc_results

    qc_results = {}

    min_refrac, max_refrac = 1e-3, 10e-3
    n_refrac = 100
    refractory_periods = np.exp(np.linspace(np.log(min_refrac), np.log(max_refrac), n_refrac))

    min_contam, max_contam = 5e-3, .35
    n_contam = 50
    contamination_test_proportions = np.exp(np.linspace(np.log(min_contam), np.log(max_contam), n_contam))

    cids = np.unique(spike_clusters)

    rvl = compute_rvl_tensor(spike_times, spike_clusters, cids, refractory_periods, contamination_test_proportions, progress=True)

    pdf = PdfPages(cache_dir / 'refractory_qc.pdf')
    for iU in tqdm(range(len(cids)) , desc='Plotting refractory QC'):
        cid = cids[iU]
        likelihoods = rvl[iU].squeeze()
        cluster_spikes = spike_times[spike_clusters == cid]
        fig, axs = plot_rvl(cluster_spikes, likelihoods, refractory_periods, contamination_test_proportions)
        axs[0].set_title(f'Cluster {cid}\nISI Distribution')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

    qc_results = {'rvl_tensor': rvl, 'refractory_periods': refractory_periods, 'contamination_test_proportions': contamination_test_proportions}
    np.savez(cache_dir / 'refractory_qc.npz', **qc_results)
    return qc_results

def waveform_qc(seg, spike_samples, spike_clusters, cache_dir, n_waves=512, n_samples=82, uV_per_bit=0.195, recalc=False):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'waveforms.npz'
    
    if npz_path.exists() and not recalc:
        waveforms = np.load(npz_path)
        return waveforms


    cids = np.unique(spike_clusters)
    n_clusters = len(cids)
    n_channels = seg.get_num_channels()
    waveforms = np.zeros((n_clusters, n_samples, n_channels), np.float32)
    samples = np.zeros((n_clusters, n_waves),np.int64) - 1
    times = (np.arange(n_samples) - n_samples//2) / seg.get_sampling_frequency()
    
    
    with tqdm(total=len(cids), desc='Extracting waveforms') as pbar:
        for iC, cid in enumerate(cids):
            cluster_samples = spike_samples[spike_clusters == cid]
            n_waves_clust = np.min([n_waves, len(cluster_samples)]) 
            sub_inds = np.random.choice(len(cluster_samples), n_waves_clust, replace=False)
            cluster_samples_sub = cluster_samples[sub_inds]
            samples[iC, :n_waves_clust] = cluster_samples_sub

            traces = np.zeros((n_waves_clust, n_samples, seg.get_num_channels())) 
            
            for iW, iS in enumerate(cluster_samples_sub):
                i0 = max(0, iS - n_samples // 2)
                i1 = min(seg.get_num_frames()-1, iS + (n_samples - n_samples // 2))
                wave = seg.get_traces(start_frame=i0, end_frame=i1) * uV_per_bit
                o0 = i0 - (iS - n_samples // 2)
                o1 = o0 + i1 - i0
                traces[iW, o0:o1, :] = wave
            waveforms[iC,...] = np.median(traces, axis=0)
            pbar.update(1)

    out = {'waveforms': waveforms, 'samples': samples, 'times': times, 'cids': cids}
    np.savez(npz_path, **out)
    return out

def run_qc(seg, results, cache_dir, recalc=False):
    '''
    Run the quality control pipeline on the given sorted data.
    
    Parameters
    ----------
    seg: spikeinterface recording segment
        The recording segment which was sorted. Used to extract waveforms and other data.
    results: KilosortResults
        The results of the kilosort4 sorting.
    
    Returns
    -------
    qc_results: dict
        The results of the quality control pipeline
    '''

    qc_results = {}

    spike_samples = results.spike_times

    spike_times = results.spike_times / seg.get_sampling_frequency()
    spike_clusters = results.spike_clusters
    spike_amplitudes = results.st[:, 2]

    wave_dir = cache_dir / 'waveforms'
    waveforms = waveform_qc(seg, spike_samples, spike_clusters, wave_dir, recalc=recalc)
    qc_results['waveforms'] = waveforms

    truncation_dir = cache_dir / 'amp_truncation'
    truncation, present = truncation_qc(spike_times, spike_clusters, spike_amplitudes, truncation_dir, recalc=recalc)
    qc_results['truncation'] = truncation
    qc_results['present'] = present

    refractory_dir = cache_dir / 'refractory'
    refractory = refractory_qc(spike_times, spike_clusters, refractory_dir, recalc=recalc)
    qc_results['refractory'] = refractory    

    return qc_results

def load_qc(cache_dir):
    '''
    Load the quality control results from a given directory.
    
    Parameters
    ----------
    cache_dir: str or Path
        The directory to load the quality control results from.
    
    Returns
    -------
    qc_results: dict
        The quality control results
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    qc_results = {}

    wave_dir = cache_dir / 'waveforms'
    waveforms = np.load(wave_dir / 'waveforms.npz')
    qc_results['waveforms'] = waveforms

    truncation_dir = cache_dir / 'amp_truncation'
    truncation = np.load(truncation_dir / 'truncation_qc.npz')
    present = np.load(truncation_dir / 'present_qc.npz')
    qc_results['truncation'] = truncation
    qc_results['present'] = present

    refractory_dir = cache_dir / 'refractory'
    refractory = np.load(refractory_dir / 'refractory_qc.npz')
    qc_results['refractory'] = refractory

    return qc_results


#%%
