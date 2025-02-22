from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def truncated_sigmoid(x, x0, k, A, x_min):
    ''' 
    A sigmoid which goes from 1-A to 1 with slope k and offset x0
    '''
    return (A / (1 + np.exp(-k * (x - x0))) - A + 1) * (x > x_min)

def fit_truncated_sigmoid(x, y, x_min = 8):
    from scipy.optimize import curve_fit

    f = lambda x, x0, k, A: truncated_sigmoid(x, x0, k, A, x_min)

    x0 = np.sum(x * y) / np.sum(y) # mean amplitude
    A0 = 1 # CDF goes from 0 to 1
    k0 = 1 # slope
    p0 = [x0, k0, A0]
    bounds = ([x_min, 0, 0], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds)

    return popt

def untruncated_sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def truncated_sigmoid_missing_pct(popt, x_min=8):
    x0, k, A = popt
    return 100 * untruncated_sigmoid(x_min, x0, k)

def fit_amp_cdf(amps, x_min = None):
    amps = np.sort(amps)
    n = len(amps)
    p = np.arange(n) / n
    if x_min is None:
        x_min = np.min(amps)
    popt = fit_truncated_sigmoid(amps, p, x_min)
    missing_pct = truncated_sigmoid_missing_pct(popt, x_min)
    return popt, missing_pct

def construct_windows(ts, max_isi, spikes_per_window):
    n_spikes = len(ts)
    dts = np.diff(ts)
    blocks = np.stack([
                np.concatenate([[0], np.where(dts > max_isi)[0] + 1]),
                np.concatenate([np.where(dts > max_isi)[0], [n_spikes-1]])
            ], axis=1)
    n_windows = len(blocks)
    valid_blocks = []
    window_blocks = []
    window_block_times = []
    for iW in range(n_windows):
        i0, i1 = blocks[iW]
        n_samples = i1 - i0 + 1
        n_windows = n_samples // spikes_per_window
        n_window_samples = spikes_per_window * n_windows
        if n_windows == 0:
            continue
        # equally space windows in the block centered in the middle
        for iB in range(i0 + (n_samples - n_window_samples) // 2, 
                        i0 + (n_samples - n_window_samples) // 2 + n_window_samples-1,
                        spikes_per_window):
            window_blocks.append((iB, iB + spikes_per_window-1))
            window_block_times.append((ts[iB], ts[iB + spikes_per_window-1]))
        valid_blocks.append((i0 + n_samples // 2 - n_window_samples // 2, 
                             i0 + n_samples // 2 - n_window_samples // 2 + n_window_samples))
    window_blocks = np.array(window_blocks)
    valid_blocks = np.array(valid_blocks)

    return window_blocks, valid_blocks

def analyze_amplitude_truncation(spike_times, spike_amplitudes, max_isi = 10, spikes_per_window = 1000):
    window_blocks, valid_blocks = construct_windows(spike_times, max_isi, spikes_per_window)

    mpcts = np.zeros(len(window_blocks))
    popts = np.zeros((len(window_blocks), 3))
    for iB, (i0, i1) in enumerate(window_blocks):
        amps = spike_amplitudes[i0:i1]
        popts[iB], mpcts[iB] = fit_amp_cdf(amps)
    
    return window_blocks, valid_blocks, popts, mpcts

def plot_amplitude_truncation(spike_times, spike_amplitudes, window_blocks, valid_blocks, mpcts):
    window_block_times = np.array([[spike_times[i0], spike_times[i1]] for i0, i1 in window_blocks])
    if window_block_times.ndim == 1:
        window_block_times = window_block_times[np.newaxis, :]

    valid_mask = np.zeros(len(spike_times), dtype=bool)
    for i0, i1 in valid_blocks:
        valid_mask[i0:i1] = True

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].hist2d(spike_times, spike_amplitudes, bins=(200, 50), cmap='Blues')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude (a.u.)')
    axs[0].set_title(f'Amplitude vs Time')
    if len(window_block_times) > 1:
        axs[1].bar(np.mean(window_block_times, axis=1), mpcts,
            width=np.diff(window_block_times, axis=1).squeeze(), 
            edgecolor='C0', color='C0', alpha=.7)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Missing %')
    axs[1].set_title('Missing Percentage from Fit')
    axs[1].fill_between(spike_times, 0, 55, where=~valid_mask, color='r', alpha=.25)
    axs[1].set_ylim(0, 55)
    plt.tight_layout()
    return fig, axs
