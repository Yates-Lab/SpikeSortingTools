#%%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from spikeinterface.preprocessing import blank_staturation
from spikeinterface.preprocessing import phase_shift
from spikeinterface.preprocessing import interpolate_bad_channels
from scipy.signal import medfilt, welch
from spikeinterface.preprocessing import highpass_filter
from spikeinterface.preprocessing import common_reference
from spikeinterface.preprocessing import highpass_filter

from probeinterface import Probe
from probeinterface.plotting import plot_probe

import os

def get_45mm_npx_probe(debug=False):
    npx_nhp_positions = np.zeros((384, 2))
    npx_nhp_row_pitch = 20
    npx_nhp_col_pitch = 103
    npx_nhp_width = 12 # 12x12um
    for iR in range(192):
        npx_nhp_positions[2*iR, 0] = 0
        npx_nhp_positions[2*iR, 1] = npx_nhp_row_pitch * iR
        npx_nhp_positions[2*iR+1, 0] = npx_nhp_col_pitch
        npx_nhp_positions[2*iR+1, 1] = npx_nhp_row_pitch * iR

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=npx_nhp_positions, shapes='square', shape_params={'width': npx_nhp_width})
    probe.set_device_channel_indices(np.arange(384))
    if debug:
        plot_probe(probe)
    return probe

def get_default_job_kwargs():
    n_cpus = os.cpu_count()
    n_cpus = n_cpus if n_cpus is not None else 1
    n_jobs = max(1, n_cpus - 1) 
    job_kwargs = dict(n_jobs=n_jobs, 
                      chunk_duration='2s', 
                      progress_bar=True,)
    return job_kwargs

def get_channel_metrics(seg, n_batches=50, batch_duration=2, med_n=11, psd_cuttoff=.8, psd_n_samples=2048, uV_per_bit=.195, freq_min=300, seed = 1002, debug=0):
    '''
    Compute channel metrics for a recording segment

    Parameters
    ----------
    seg : RecordingExtractor
        Recording segment
    n_batches : int, optional
        Number of batches to compute metrics, by default 50
    batch_duration : int, optional
        Duration of batches in seconds, by default 2
    med_n : int, optional
        Window size for median filter, by default 11
    psd_cuttoff : float, optional
        Fraction of Nyquist frequency to compute noise power, by default .8
    psd_n_samples : int, optional
        Number of samples for PSD, by default 2048
    uV_per_bit : float, optional
        Microvolts per bit, by default .195
    freq_min : int, optional
        Minimum frequency for highpass filter, by default 300
    seed : int, optional
        Random seed, by default 1002
    debug : int, optional
        Debug flag, by default 0

    Returns
    -------
    similarity : np.ndarray
        Similarity to median for each channel
    noise : np.ndarray
        Noise power for each channel

    '''
    seg = highpass_filter(seg, freq_min=freq_min, direction='forward-backward')

    fs = seg.get_sampling_frequency()
    n_samples = seg.get_num_frames()
    n_channels = seg.get_num_channels()
    batch_size = int(batch_duration * fs)
    f_thresh = psd_cuttoff * fs / 2
    similarity = np.zeros(n_channels)
    noise = np.zeros(n_channels)
    batches = np.arange(0, n_samples//batch_size*batch_size, batch_size)
    np.random.seed(seed)
    batch_sub = np.random.choice(batches, n_batches, replace=False)
    iter = tqdm(batch_sub, desc='Computing channel metrics') if debug else batch_sub
    for iB in iter:
        iE = min(iB + batch_size, n_samples)
        trace = seg.get_traces(start_frame=iB, end_frame=iE) * uV_per_bit
        med = np.median(trace, axis=1)
        med_e = np.sum(med**2)
        cc = np.sum(trace * med[:, None], axis=0) / med_e
        cc_detrend = cc - medfilt(cc, med_n)
        similarity += cc_detrend

        f, psd = welch(trace, fs=fs, nperseg=psd_n_samples, axis=0)
        noise += np.mean(psd[f > f_thresh], axis=0)

    similarity /= n_batches
    noise /= n_batches
    return similarity, noise


def condition_signal(seg, cache_dir, recalc=False, uV_per_bit=.195, uV_thresh=.5e3, similarity_thresh=-0.5, noise_thresh=1e-2, job_kwargs={}):

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print('\tPreprocessing...')

    job_kwargs = dict(get_default_job_kwargs(), **job_kwargs)

    # Add neuropixels probe if not already present
    n_channels = seg.get_num_channels()
    # if (n_channels == 384 or n_channels == 385) and not seg.has_probe():
    #     print('\tProbe metadata missing. Adding default Neuropixels NHP long configuration...')
    #     probe = get_45mm_npx_probe()
    #     seg = seg.set_probe(probe)
    #     inter_sample_shift = np.tile(np.repeat(np.arange(12) / 13, 2), 16)
    #     seg.set_property('inter_sample_shift', inter_sample_shift)

    seg_shift = phase_shift(seg)
    seg_sat = blank_staturation(seg_shift, uV_thresh / uV_per_bit, direction='both') #remove blanks? Previously this was before the phase shift?

     
     
    f_cm = cache_dir / 'channel_metrics.npy'
    if not f_cm.exists() or recalc:
        similarity, noise = get_channel_metrics(seg_sat, n_batches=50, debug=True)
        if cache_dir is not None:
            np.save(cache_dir / 'channel_metrics.npy', np.stack((similarity, noise)))
    else:
        similarity, noise = np.load(cache_dir / 'channel_metrics.npy')

    noisy_channels = noise > noise_thresh
    dead_channels = similarity < similarity_thresh
    bad_channels = np.logical_or(noisy_channels, dead_channels)
    print(f'\tFound {np.sum(noisy_channels)} noisy channels and {np.sum(dead_channels)} dead channels')

    ids = seg_sat.get_channel_ids()
    bad_ids = ids[bad_channels]
    seg_interp = interpolate_bad_channels(seg_sat, bad_ids)

    seg_cr = common_reference(seg_interp, reference = 'global', operator = 'median')

    seg_hp = highpass_filter(seg_cr, freq_min=300., direction='forward-backward')
    
    fig, axs = plt.subplots(1,2, figsize=(8,6), sharey=True)
    axs[0].plot(similarity, np.arange(n_channels))
    axs[0].scatter(similarity[dead_channels], np.where(dead_channels)[0], color='r')
    axs[0].axvline(similarity_thresh, color='r', linestyle='--')
    axs[0].set_title('Similarity to Median')
    axs[0].set_xlabel('Similarity')
    axs[0].set_ylabel('Channel')
    axs[1].plot(noise, np.arange(n_channels))
    axs[1].scatter(noise[noisy_channels], np.where(noisy_channels)[0], color='r')
    axs[1].axvline(noise_thresh, color='r', linestyle='--')
    axs[1].set_title('Noise Power (>.8 Nyquist)')
    axs[1].set_xlabel('Power (uV^2/Hz)')
    plt.tight_layout()
    fig.savefig(cache_dir / 'channel_metrics.png')
    plt.close('all')

    return seg_hp


