#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .preprocess import get_default_job_kwargs
import medicine
from medicine.plotting import _correct_motion_on_peaks, plot_motion_correction
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, motion_utils, interpolate_motion
from spikeinterface.preprocessing import astype
from scipy.signal import medfilt

from spikeinterface.core.motion import Motion

def correct_motion(seg, cache_dir, detect_peak_args={}, localize_peak_args={}, ks_motion_args={}, dc_motion_args={}, med_motion_args={}, job_kwargs={}, recalc=False, method='med', median_filter_size=1):

    print('Starting motion correction...')


    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    job_kwargs = dict(get_default_job_kwargs(), **job_kwargs)

    ###
    # Detect peaks
    ###

    default_detect_peak_args = dict(
        method = 'locally_exclusive',  #'locally_exclusive', # replace with locally_exclusive_torch to use DetectPeakLocallyExclusiveTorch ???
        radius_um = 50, #was 100, possibly for the nhp probes, resetting to default 
        detect_threshold=5 #7
    )
    detect_peak_args = dict(default_detect_peak_args, **detect_peak_args)

    f_peaks = cache_dir / 'peaks.npy'
    if not f_peaks.exists() or recalc:
        peaks = detect_peaks(seg, **detect_peak_args, **job_kwargs)
        np.save(cache_dir / 'peaks.npy', peaks)
    else:
        peaks = np.load(f_peaks)

    ###
    # Localize peaks
    ###

    default_localize_peak_args = dict(method = 'monopolar_triangulation')
    localize_peak_args = dict(default_localize_peak_args, **localize_peak_args)

    f_peak_locations = cache_dir / 'peak_locations.npy'
    if not f_peak_locations.exists() or recalc:
        peak_locations = localize_peaks(seg, peaks, **localize_peak_args, **job_kwargs)
        np.save(cache_dir / 'peak_locations.npy', peak_locations)
    else:
        peak_locations = np.load(f_peak_locations)

    ###
    # Kilosort motion
    ###

    if method == 'ks' or method == 'all':
        print('Estimating Kilosort-like motion...')

        default_ks_motion_args = dict(method = 'iterative_template', direction = 'y', bin_s = 2.0, num_shifts_block = 5)
        ks_motion_args = dict(default_ks_motion_args, **ks_motion_args)
        ks_motion_args['method'] = 'iterative_template'

        ks_motion_dir = cache_dir / 'ks-motion'
        ks_motion_dir.mkdir(parents=True, exist_ok=True)
        if not (ks_motion_dir / "motion.npy").exists() or recalc:
            ks_motion = estimate_motion(
                recording = seg, 
                peaks = peaks,
                peak_locations = peak_locations,
                **ks_motion_args 
            )
            ks_displacement = ks_motion.displacement[0]
            if median_filter_size > 1:
                ks_displacement = medfilt(ks_displacement, kernel_size=(median_filter_size, 1))

            np.save(ks_motion_dir / "motion.npy", ks_displacement)
            np.save(ks_motion_dir / "time_bins.npy", ks_motion.temporal_bins_s[0])
            np.save(ks_motion_dir / "depth_bins.npy", ks_motion.spatial_bins_um)

        # load kilosort motion
        ks_motion = Motion(
            displacement=np.load(ks_motion_dir / "motion.npy"),
            temporal_bins_s=np.load(ks_motion_dir / "time_bins.npy"),
            spatial_bins_um=np.load(ks_motion_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = ks_motion
        
        
    ###
    # Varol2021 decentralized motion
    ###
    if method == 'dc' or method == 'all':
        print('Estimating decentralized motion...')

        default_dc_motion_args = dict(method = 'decentralized', direction = 'y', bin_s = 2.0)
        dc_motion_args = dict(default_dc_motion_args, **dc_motion_args)

        decentralized_motion_dir = cache_dir / 'decentralized-motion'
        decentralized_motion_dir.mkdir(parents=True, exist_ok=True)
        if not (decentralized_motion_dir / "motion.npy").exists() or recalc:
            dc_motion = estimate_motion(
                recording = seg, 
                peaks = peaks,
                peak_locations = peak_locations,
                **dc_motion_args
            )
            dc_displacement = dc_motion.displacement[0]
            if median_filter_size > 1:
                dc_displacement = medfilt(dc_displacement, kernel_size=(median_filter_size, 1))
            np.save(decentralized_motion_dir / "motion.npy", dc_displacement)
            np.save(decentralized_motion_dir / "time_bins.npy", dc_motion.temporal_bins_s[0])
            np.save(decentralized_motion_dir / "depth_bins.npy", dc_motion.spatial_bins_um)

        # load decentralized motion
        dc_motion = Motion(
            displacement=np.load(decentralized_motion_dir / "motion.npy"),
            temporal_bins_s=np.load(decentralized_motion_dir / "time_bins.npy"),
            spatial_bins_um=np.load(decentralized_motion_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = dc_motion

    ###
    # MEDiCINe motion
    ###
    if method == 'med' or method == 'all':
        print('Estimating MEDiCINe motion...')

        default_med_motion_args = dict(time_bin_size = 2.0, num_depth_bins = 2)
        med_motion_args = dict(default_med_motion_args, **med_motion_args)

        # Create directory to store MEDiCINe outputs for this recording
        medicine_output_dir = cache_dir / 'medicine'
        medicine_output_dir.mkdir(parents=True, exist_ok=True)
        if not (medicine_output_dir / "motion.npy").exists() or recalc:
            medicine.run_medicine(
                peak_amplitudes=peaks['amplitude'],
                peak_depths=peak_locations['y'],
                peak_times=peaks['sample_index'] / seg.get_sampling_frequency() + seg.get_time_info()['t_start'],
                output_dir=medicine_output_dir,
                **med_motion_args
            )

            # Load MEDiCINe outputs
            med_motion = np.load(medicine_output_dir / "motion.npy")
            med_time_bins = np.load(medicine_output_dir / "time_bins.npy")
            med_depth_bins = np.load(medicine_output_dir / "depth_bins.npy")
            n_append = 5
            dt = med_time_bins[1] - med_time_bins[0]
            med_time_bins = np.concatenate(
                    [med_time_bins, med_time_bins[-1] + np.arange(1, n_append + 1) * dt]
            )
            med_motion = np.concatenate(
                [med_motion, np.ones((n_append, med_motion.shape[1])) * med_motion[-1]]
            )
            if median_filter_size > 1:
                med_motion = medfilt(med_motion, kernel_size=(median_filter_size, 1))
            np.save(medicine_output_dir / "motion.npy", med_motion)
            np.save(medicine_output_dir / "time_bins.npy", med_time_bins)
            np.save(medicine_output_dir / "depth_bins.npy", med_depth_bins)

        # Load MEDiCINe outputs
        med_motion = Motion( 
            displacement=np.load(medicine_output_dir / "motion.npy"),
            temporal_bins_s=np.load(medicine_output_dir / "time_bins.npy"),
            spatial_bins_um=np.load(medicine_output_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = med_motion

    # Interpolate motion using MEDiCINe
    #motion = med_motion
    #if method == 'ks':
    #    motion = ks_motion
    #if method == 'dc':
    #    motion = dc_motion

    seg_sort = astype(interpolate_motion(astype(seg, "float"), motion, border_mode='force_zeros'), "int16")

    print('Finished motion correction')
    return seg_sort

def plot_motion_output(seg, cache_dir, save_dir=None, plot_stride=30, uV_per_bit=.195, recalc=False):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    if save_dir is not None and isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir is None:
        save_dir = cache_dir

    save_files = [
        'depth_raster.png',
        'motion_comparison.png',
        'amplitude_depth_comparison.png',
        'kilosort_motion_correction.png',
        'decentralized_motion_correction.png',
        'medicine_motion_correction.png',
    ]
    if all([(save_dir / f).exists() for f in save_files]) and not recalc:
        print('All plots already exist, returning...')
        return

    peaks = np.load(cache_dir / 'peaks.npy')
    peak_locations = np.load(cache_dir / 'peak_locations.npy')
    ks_loc= (cache_dir / 'ks-motion')
    dc_loc = (cache_dir / 'decentralized-motion')
    med_loc = (cache_dir / 'medicine')
    if ks_loc.exists():
        ks_motion = Motion(
            displacement=np.load(cache_dir / "ks-motion/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "ks-motion/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "ks-motion/depth_bins.npy"),
        )
        method = 'ks'
    if dc_loc.exists():
        dc_motion = Motion(
            displacement=np.load(cache_dir / "decentralized-motion/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "decentralized-motion/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "decentralized-motion/depth_bins.npy"),
        )
        method = 'dc'
    if med_loc.exists():
        med_motion = Motion(
            displacement=np.load(cache_dir / "medicine/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "medicine/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "medicine/depth_bins.npy"),
        )
        method = 'med'
    if ks_loc.exists() and dc_loc.exists() and med_loc.exists():
        method = 'all'
    

    spike_samples = peaks['sample_index']
    spike_times = spike_samples / seg.get_sampling_frequency() + seg.get_time_info()['t_start']
    spike_depths = peak_locations['y']
    spike_amps = peaks['amplitude'] * uV_per_bit

    # Subsample
    peak_samples = spike_samples[::plot_stride]
    peak_times = spike_times[::plot_stride]
    peak_depths = spike_depths[::plot_stride]
    peak_amplitudes = spike_amps[::plot_stride]

    # Normalize amplitudes by CDF to have uniform distribution
    amp_argsort = np.argsort(np.argsort(peak_amplitudes))
    peak_amplitudes = amp_argsort / len(peak_amplitudes)

    #
    # Plot depth raster
    #
    # Function for plotting neural activity
    def _plot_neural_activity(ax, times, depths, colors):
        plot = ax.scatter(times, depths, s=1, c=colors, alpha=.75)
        ax.set_xlabel("time (s)", fontsize=12)
        ax.set_ylabel("depth from probe tip (um)", fontsize=12)
        return plot

    # Scatterplot peaks
    cmap = plt.get_cmap("winter")
    colors = cmap(peak_amplitudes)
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    plot = _plot_neural_activity(axs, peak_times, peak_depths, colors)
    fig.colorbar(plot, ax=axs)
    fig.savefig(save_dir / 'depth_raster.png')

    #
    # Plot motion estimate comparison
    #

    depth = med_motion.spatial_bins_um[0]
    times = med_motion.temporal_bins_s[0]

    probe = seg.get_probe()
    d_min = np.min(probe.contact_positions[:, 1])
    d_max = np.max(probe.contact_positions[:, 1])
    n_depths = 5
    depths = np.linspace(d_min, d_max, n_depths)
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True) 
    ks_motion_depths = np.zeros((len(times), n_depths))
    dc_motion_depths = np.zeros((len(times), n_depths))
    med_motion_depths = np.zeros((len(times), n_depths))

    for i, depth in enumerate(depths):

        dist = (d_max - depth)
        if method == 'ks' or method == 'all':
            ks_motion_interp = ks_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            ks_motion_depths[:,n_depths-i-1] = ks_motion_interp
            axs[i].plot(times, ks_motion_interp, label='Kilosort')
        
        if method == 'dc' or method == 'all':
            dc_motion_interp = dc_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            dc_motion_depths[:,n_depths-i-1] = dc_motion_interp
            axs[i].plot(times, dc_motion_interp, label='Decentralized')

        if method == 'med' or method == 'all':
            med_motion_interp = med_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            med_motion_depths[:,n_depths-i-1] = med_motion_interp
            axs[i].plot(times, med_motion_interp, label='MEDiCINe')


        if i == n_depths // 2: 
            axs[i].set_ylabel('Motion (um)')
        if i == n_depths - 1:
            axs[i].set_xlabel('Time (s)')
        axs[i].set_title(f'Motion estimates (depth = {depth} um)')
        if i == 0:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'motion_comparison.png')

    #
    # Plot amplitude-depth comparison
    #
    # Get colors and create figure
    cmap = plt.get_cmap('winter')
    colors = cmap(peak_amplitudes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)

    if method == 'ks' or method == 'all':
        peak_depth_ks = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            ks_motion_depths,
            times,
            depths
        )

        _ = _plot_neural_activity(axes[0], peak_times, peak_depth_ks, colors)
        axes[0].set_title("Kilosort")

    if method == 'dc' or method == 'all':
        peak_depth_dc = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            dc_motion_depths,
            times,
            depths
        )
        _ = _plot_neural_activity(axes[1], peak_times, peak_depth_dc, colors)
        axes[1].set_title("Decentralized")

    if method == 'med' or method == 'all':
        peak_depth_med = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            med_motion_depths,
            times,
            depths
        )
        plot = _plot_neural_activity(axes[2], peak_times, peak_depth_med, colors)
        axes[2].set_title("MEDiCINe")
        fig.colorbar(plot, ax=axes[2]) 
    plt.tight_layout()
    plt.savefig(save_dir / 'amplitude_depth_comparison.png')

    #
    #   Plot individual motion correction
    #

    # Kilosort
    if method == 'ks' or method == 'all':
        f_ks = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            ks_motion_depths,
        )
        f_ks.suptitle('Kilosort')
        f_ks.savefig(save_dir / 'kilosort_motion_correction.png')

    # Decentralized
    if method == 'dc' or method == 'all':
        f_dc = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            dc_motion_depths,
        )
        f_dc.suptitle('Decentralized')    
        f_dc.savefig(save_dir / 'decentralized_motion_correction.png')

    # MEDiCINe
    if method == 'med' or method == 'all':
        f_med = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            med_motion_depths,
        )
        f_med.suptitle('MEDiCINe')
        f_med.savefig(save_dir / 'medicine_motion_correction.png')
    
    plt.close('all')


