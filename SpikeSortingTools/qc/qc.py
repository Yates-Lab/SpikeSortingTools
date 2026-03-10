#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .refractory import compute_min_contam_props_analytical, plot_min_contam_prop
from .truncation import analyze_amplitude_truncation, plot_amplitude_truncation
from pathlib import Path
from tqdm import tqdm
import json


def truncation_qc(spike_times, spike_clusters, spike_amplitudes, cache_dir, max_isi=np.inf, spikes_per_window=1000, recalc=False):
    '''
    Run the truncation quality control analysis on the given sorted data.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster ID for each spike.
    spike_amplitudes : array-like (n_spikes,)
        Amplitude for each spike.
    cache_dir : str or Path
        Directory to cache results.
    max_isi : float
        Maximum ISI for amplitude truncation analysis.
    spikes_per_window : int
        Number of spikes per analysis window.
    recalc : bool
        Force recalculation.

    Returns
    -------
    trunc_qc : dict
        Truncation QC results.
    pres_qc : dict
        Presence QC results.
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    truncation_path = cache_dir / 'truncation.npz'
    present_path = cache_dir / 'present.npz'

    if truncation_path.exists() and present_path.exists() and not recalc:
        trunc_qc = np.load(truncation_path)
        pres_qc = np.load(present_path)
        return trunc_qc, pres_qc

    cids = np.unique(spike_clusters)

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

    for cid in tqdm(cids, desc='Running truncation QC'):
        cluster_spikes = spike_times[spike_clusters == cid]
        cluster_amps = spike_amplitudes[spike_clusters == cid]
        window_blocks, valid_blocks, popts, mpcts = analyze_amplitude_truncation(cluster_spikes, cluster_amps, max_isi=max_isi, spikes_per_window=spikes_per_window)

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

    trunc_qc = {k: np.concatenate(v, axis=0) for k, v in trunc_qc.items()}
    pres_qc = {k: np.concatenate(v, axis=0) for k, v in pres_qc.items()}

    np.savez(truncation_path, **trunc_qc)
    np.savez(present_path, **pres_qc)

    return trunc_qc, pres_qc


def generate_truncation_pdf(spike_times, spike_clusters, spike_amplitudes, trunc_qc, pres_qc, pdf_path):
    '''
    Generate a standalone PDF with per-unit amplitude truncation plots.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster ID for each spike.
    spike_amplitudes : array-like (n_spikes,)
        Amplitude for each spike.
    trunc_qc : dict
        Truncation QC results (from truncation_qc or load_qc).
    pres_qc : dict
        Presence QC results (from truncation_qc or load_qc).
    pdf_path : str or Path
        Path to save the output PDF.
    '''
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cids = np.unique(spike_clusters)
    with PdfPages(pdf_path) as pdf:
        for cid in tqdm(cids, desc='Plotting truncation QC'):
            st_clu = spike_times[spike_clusters == cid]
            amp_clu = spike_amplitudes[spike_clusters == cid]
            trunc_mask = trunc_qc['cid'] == cid
            window_blocks = trunc_qc['window_blocks'][trunc_mask]
            mpcts = trunc_qc['mpcts'][trunc_mask]
            valid_mask = pres_qc['cid'] == cid
            valid_blocks = pres_qc['valid_blocks'][valid_mask]
            fig, axs = plot_amplitude_truncation(st_clu, amp_clu, window_blocks, valid_blocks, mpcts)
            axs[0].set_title(f'Cluster {cid}\nAmplitudes vs Time')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def refractory_qc(spike_times, spike_clusters, cache_dir, min_refractory_period=1e-3, max_refractory_period=10e-3, n_refractory_periods=100, recalc=False):
    '''
    Run the refractory period quality control analysis on the given sorted data.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        The cluster assignments of each spike.
    cache_dir : str or Path
        Directory to cache results.
    min_refractory_period : float
        Minimum refractory period to test (seconds).
    max_refractory_period : float
        Maximum refractory period to test (seconds).
    n_refractory_periods : int
        Number of refractory periods to test.
    recalc : bool
        Force recalculation.

    Returns
    -------
    qc_results : dict
        The refractory QC results.
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'refractory.npz'
    if npz_path.exists() and not recalc:
        qc_results = np.load(npz_path)
        return qc_results

    refractory_periods = np.exp(np.linspace(np.log(min_refractory_period), np.log(max_refractory_period), n_refractory_periods))

    min_contam_props, firing_rates = compute_min_contam_props_analytical(spike_times, spike_clusters, refractory_periods=refractory_periods, progress=True)

    qc_results = {'min_contam_props': min_contam_props, 'refractory_periods': refractory_periods, 'firing_rates': firing_rates}
    np.savez(npz_path, **qc_results)
    return qc_results


def generate_refractory_pdf(spike_times, spike_clusters, refractory_results, pdf_path):
    '''
    Generate a standalone PDF with per-unit refractory period violation plots.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster ID for each spike.
    refractory_results : dict
        Refractory QC results (from refractory_qc or load_qc['refractory']).
    pdf_path : str or Path
        Path to save the output PDF.
    '''
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    min_contam_props = refractory_results['min_contam_props']
    refractory_periods = refractory_results['refractory_periods']
    firing_rates = refractory_results['firing_rates']
    cids = np.unique(spike_clusters)

    with PdfPages(pdf_path) as pdf:
        for iU in tqdm(range(len(cids)), desc='Plotting refractory QC'):
            cid = cids[iU]
            st_clu = spike_times[spike_clusters == cid]
            n_spikes = len(st_clu)
            fig, axs = plot_min_contam_prop(st_clu, min_contam_props[iU], refractory_periods)
            axs.set_title(f'Cluster {cid} - {n_spikes} spikes ({firing_rates[iU]:.2f} Hz)')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

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



def generate_qc_report(qc_results, spike_times, spike_clusters, spike_amplitudes, pdf_path, cids=None):
    '''
    Generate a combined QC PDF report with one page per unit showing waveform,
    amplitude truncation, and refractory period violation plots.

    Parameters
    ----------
    qc_results : dict
        QC results dictionary as returned by run_qc or load_qc.
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster ID for each spike.
    spike_amplitudes : array-like (n_spikes,)
        Amplitude for each spike.
    pdf_path : str or Path
        Path to save the output PDF.
    cids : array-like, optional
        Cluster IDs to include. If None, uses all unique clusters.
    '''
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if cids is None:
        cids = np.unique(spike_clusters)

    with PdfPages(pdf_path) as pdf:
        for cid in tqdm(cids, desc="Generating QC report"):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[.5, 1.5, 1])

            # Waveform panel
            cid_int = int(cid)
            waveform = qc_results['waveforms']['waveforms'][cid_int]
            wv_max = np.max(np.abs(waveform))
            n_channels = waveform.shape[1]
            t_min = qc_results['waveforms']['times'].min() * 1e3
            t_max = qc_results['waveforms']['times'].max() * 1e3

            im_kwarg = dict(cmap='coolwarm', aspect='auto',
                            vmin=-wv_max, vmax=wv_max, origin='lower')
            axs[0].imshow(waveform[:, ::2].T, extent=[t_min, t_max, 0, n_channels], **im_kwarg)
            axs[0].axvline(t_max, color='k', linestyle='--')
            axs[0].imshow(waveform[:, 1::2].T, extent=[t_max, 2*t_max - t_min, 0, n_channels], **im_kwarg)
            axs[0].set_xlim(t_min, 2*t_max - t_min)
            axs[0].set_xticks([t_min, t_max])
            axs[0].set_xticklabels([f'{t_min:.2f}', f'{t_max:.2f}'])
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Channel')
            axs[0].set_title(f'Cluster {cid}\nWaveform - {wv_max:.2f} uV')

            # Amplitude truncation panel
            trunc_mask = qc_results['truncation']['cid'] == cid
            st_clu = spike_times[spike_clusters == cid]
            amp_clu = spike_amplitudes[spike_clusters == cid]
            window_blocks = qc_results['truncation']['window_blocks'][trunc_mask]
            mpcts = qc_results['truncation']['mpcts'][trunc_mask]
            valid_mask = qc_results['present']['cid'] == cid
            valid_blocks = qc_results['present']['valid_blocks'][valid_mask]
            plot_amplitude_truncation(st_clu, amp_clu, window_blocks, valid_blocks, mpcts, axs=axs[1])
            axs[1].set_title(f'Cluster {cid}\nAmplitudes vs Time')

            # Refractory period panel
            min_contam_props = qc_results['refractory']['min_contam_props'][cid_int]
            refractory_periods = qc_results['refractory']['refractory_periods']
            firing_rate = qc_results['refractory']['firing_rates'][cid_int]
            plot_min_contam_prop(st_clu, min_contam_props, refractory_periods, axs=axs[2])
            axs[2].set_title(f'Cluster {cid} - {firing_rate:.2f} Hz\nMinimum Contamination Proportion')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def run_qc(seg, results, cache_dir, recalc=False, waveform_kwargs={}, truncation_kwargs={}, refractory_kwargs={}):
    '''
    Run the quality control pipeline on the given sorted data.

    This includes:
    1. Waveform extraction
    2. Amplitude truncation analysis
    3. Refractory period violation analysis
    4. Combined per-unit QC PDF report generation

    Parameters
    ----------
    seg: spikeinterface recording segment
        The recording segment which was sorted. Used to extract waveforms and other data.
    results: KilosortResults
        The results of the kilosort4 sorting.
    cache_dir: str or Path
        Directory to save QC outputs.
    recalc: bool
        If True, force recalculation of all QC metrics.

    Returns
    -------
    qc_results: dict
        The results of the quality control pipeline
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    qc_results = {}

    f_sorter_log = cache_dir / 'sorter_log.json'
    if hasattr(results, 'spikeinterface_log') and f_sorter_log.exists():
        si_log = results.spikeinterface_log
        sorter_log = json.load(open(f_sorter_log, 'r'))
        identical = True
        for k in si_log:
            if k not in sorter_log or si_log[k] != sorter_log[k]:
                identical = False
                break

        if not identical:
            print('Sorter log has changed, recalculating QC')
            recalc = True
    else:
        print('Sorter log not found, recalculating QC')
        recalc = True

    spike_samples = results.spike_samples
    spike_times = results.spike_times
    spike_clusters = results.spike_clusters
    spike_amplitudes = results.st[:, 2]

    wave_dir = cache_dir / 'waveforms'
    waveforms = waveform_qc(seg, spike_samples, spike_clusters, wave_dir, recalc=recalc, **waveform_kwargs)
    qc_results['waveforms'] = waveforms

    truncation_dir = cache_dir / 'amp_truncation'
    truncation, present = truncation_qc(spike_times, spike_clusters, spike_amplitudes, truncation_dir, recalc=recalc, **truncation_kwargs)
    qc_results['truncation'] = truncation
    qc_results['present'] = present

    refractory_dir = cache_dir / 'refractory'
    refractory = refractory_qc(spike_times, spike_clusters, refractory_dir, recalc=recalc, **refractory_kwargs)
    qc_results['refractory'] = refractory

    if recalc and hasattr(results, 'spikeinterface_log'):
        with open(f_sorter_log, 'w') as f:
            json.dump(results.spikeinterface_log, f)

    # Reload QC data as plain dicts (npz files return NpzFile objects)
    qc_results = load_qc(cache_dir)

    # Generate combined per-unit QC PDF report
    pdf_path = cache_dir / 'qc.pdf'
    if not pdf_path.exists() or recalc:
        generate_qc_report(qc_results, spike_times, spike_clusters, spike_amplitudes, pdf_path)

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
    waveforms = dict(np.load(wave_dir / 'waveforms.npz'))
    qc_results['waveforms'] = waveforms

    truncation_dir = cache_dir / 'amp_truncation'
    truncation = dict(np.load(truncation_dir / 'truncation.npz'))
    present = dict(np.load(truncation_dir / 'present.npz'))
    qc_results['truncation'] = truncation
    qc_results['present'] = present

    refractory_dir = cache_dir / 'refractory'
    refractory = dict(np.load(refractory_dir / 'refractory.npz'))
    qc_results['refractory'] = refractory

    return qc_results
