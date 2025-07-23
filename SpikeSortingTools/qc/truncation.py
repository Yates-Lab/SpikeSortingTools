from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def truncated_sigmoid(x, x0, k, A, x_min):
    ''' 
    A sigmoid function that is truncated below x_min and scaled to go from 1-A to 1.
    
    Parameters
    ----------
    x : array-like
        Input values
    x0 : float
        Offset parameter (center of sigmoid)
    k : float 
        Slope parameter (steepness of sigmoid)
    A : float
        Scale parameter (controls range from 1-A to 1)
    x_min : float
        Truncation threshold below which function returns 0
    
    Returns
    -------
    array-like
        Truncated and scaled sigmoid values
    '''
    return (A / (1 + np.exp(-k * (x - x0))) - A + 1) * (x > x_min)

def fit_truncated_sigmoid(x, y, x_min = 8):
    '''
    Fit a truncated sigmoid to data using curve_fit.
    
    Parameters
    ----------
    x : array-like
        Independent variable data
    y : array-like 
        Dependent variable data
    x_min : float, optional
        Truncation threshold, default 8
        
    Returns
    -------
    array-like
        Optimal parameters [x0, k, A]
    '''
    from scipy.optimize import curve_fit

    f = lambda x, x0, k, A: truncated_sigmoid(x, x0, k, A, x_min)

    # Initial parameter estimates
    x0 = np.sum(x * y) / np.sum(y) # mean amplitude
    A0 = 1 # CDF goes from 0 to 1
    k0 = 1 # slope
    p0 = [x0, k0, A0]
    bounds = ([x_min, 0, 0], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds)

    return popt

def untruncated_sigmoid(x, x0, k):
    '''
    Standard sigmoid function without truncation or scaling.
    '''
    return 1 / (1 + np.exp(-k * (x - x0)))

def truncated_sigmoid_missing_pct(popt, x_min=8):
    '''
    Calculate percentage of data missing below truncation threshold.
    
    Parameters
    ----------
    popt : array-like
        Parameters [x0, k, A] from fitted truncated sigmoid
    x_min : float, optional
        Truncation threshold, default 8
        
    Returns
    -------
    float
        Percentage of data estimated to be missing
    '''
    x0, k, A = popt
    return 100 * untruncated_sigmoid(x_min, x0, k)

def fit_amp_cdf(amps, x_min = None):
    '''
    Fit truncated sigmoid to empirical CDF of amplitude data.
    
    Parameters
    ----------
    amps : array-like
        Amplitude values
    x_min : float, optional
        Truncation threshold. If None, uses minimum amplitude
        
    Returns
    -------
    tuple
        (fitted parameters, estimated missing percentage)
    '''
    amps = np.sort(amps)
    n = len(amps)
    p = np.arange(n) / n  # empirical CDF
    if x_min is None:
        x_min = np.min(amps)
    popt = fit_truncated_sigmoid(amps, p, x_min)
    missing_pct = truncated_sigmoid_missing_pct(popt, x_min)
    return popt, missing_pct

def construct_windows(ts, max_isi, spikes_per_window):
    '''
    Divide spike times into windows based on inter-spike intervals.
    
    Parameters
    ----------
    ts : array-like
        Spike times
    max_isi : float
        Maximum inter-spike interval for continuous activity
    spikes_per_window : int
        Number of spikes per analysis window
        
    Returns
    -------
    tuple
        (window boundaries, valid block boundaries)
    '''
    n_spikes = len(ts)
    dts = np.diff(ts)
    # Find continuous blocks of activity separated by large ISIs
    blocks = np.stack([
                np.concatenate([[0], np.where(dts > max_isi)[0] + 1]),
                np.concatenate([np.where(dts > max_isi)[0], [n_spikes-1]])
            ], axis=1)
    n_windows = len(blocks)
    valid_blocks = []
    window_blocks = []
    window_block_times = []
    
    # Process each continuous block
    for iW in range(n_windows):
        i0, i1 = blocks[iW]
        n_samples = i1 - i0 + 1
        n_windows = n_samples // spikes_per_window
        n_window_samples = spikes_per_window * n_windows
        if n_windows == 0:
            continue
        
        # Create equally spaced windows centered in the block
        start_idx = i0 + (n_samples - n_window_samples) // 2
        for iB in range(start_idx, 
                       start_idx + n_window_samples-1,
                       spikes_per_window):
            window_blocks.append((iB, iB + spikes_per_window-1))
            window_block_times.append((ts[iB], ts[iB + spikes_per_window-1]))
        valid_blocks.append((i0 + n_samples // 2 - n_window_samples // 2, 
                           i0 + n_samples // 2 - n_window_samples // 2 + n_window_samples))
    
    return np.array(window_blocks), np.array(valid_blocks)

def analyze_amplitude_truncation(spike_times, spike_amplitudes, max_isi = 60, spikes_per_window = 1000):
    '''
    Analyze amplitude truncation across time windows.
    
    Parameters
    ----------
    spike_times : array-like
        Spike timing data
    spike_amplitudes : array-like
        Spike amplitude data
    max_isi : float, optional
        Maximum inter-spike interval, default 10
    spikes_per_window : int, optional
        Spikes per analysis window, default 1000
        
    Returns
    -------
    tuple
        (window boundaries, valid blocks, fitted parameters, missing percentages)
    '''
    window_blocks, valid_blocks = construct_windows(spike_times, max_isi, spikes_per_window)

    mpcts = np.zeros(len(window_blocks))
    popts = np.zeros((len(window_blocks), 3))
    for iB, (i0, i1) in enumerate(window_blocks):
        amps = spike_amplitudes[i0:i1]
        popts[iB], mpcts[iB] = fit_amp_cdf(amps)
    
    return window_blocks, valid_blocks, popts, mpcts

def plot_amplitude_truncation(spike_times, spike_amplitudes, 
                              window_blocks, valid_blocks, mpcts, axs=None):
    '''
    Create visualization of amplitude truncation analysis.
    
    Parameters
    ----------
    spike_times : array-like
        Spike timing data
    spike_amplitudes : array-like
        Spike amplitude data
    window_blocks : array-like
        Window boundary indices
    valid_blocks : array-like
        Valid block boundary indices
    mpcts : array-like
        Missing percentages for each window
        
    Returns
    -------
    tuple
        (figure handle, axes handles)
    '''
    window_block_times = np.array([[spike_times[i0], spike_times[i1]] for i0, i1 in window_blocks])
    if window_block_times.ndim == 1:
        window_block_times = window_block_times[np.newaxis, :]

    # Create mask for valid regions
    valid_mask = np.zeros(len(spike_times), dtype=bool)
    for i0, i1 in valid_blocks:
        valid_mask[i0:i1] = True

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = axs.get_figure()
    
    # Plot amplitude vs time
    xlim = [spike_times[0], spike_times[-1]]
    axs.hist2d(spike_times, spike_amplitudes, bins=(200, 50), cmap='Blues')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Amplitude (a.u.)')
    axs.set_title(f'Amplitude vs Time')
    axs.set_xlim(xlim)

    axs2 = axs.twinx()
    for pct, (i0, i1) in zip(mpcts, window_blocks):
        axs2.plot([spike_times[i0], spike_times[i1]], [pct, pct], color='red', linewidth=3.5)

    axs2.set_ylim([0, 52])
    axs2.set_yticks([0, 50])
    axs2.set_yticklabels(['0%', '50%'])

    axs2.fill_between(spike_times, 0, 55, where=~valid_mask, color='r', alpha=.25)
    axs2.tick_params(axis='y', colors='red')
    axs2.set_ylabel('Missing %', color='red')
    return fig, [axs, axs2]

