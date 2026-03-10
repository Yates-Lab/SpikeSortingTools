"""
Sliding Refractory Period (RP) Quality Metric
==============================================

Author: Ryan A. Ressmeyer (with strong contribution from Claude Code)

This module implements a statistical quality metric for evaluating whether spike-sorted
neural units have contaminated refractory periods. A contaminated refractory period
indicates that spikes from other neurons or noise are being mis-attributed to the unit.
These methods were adapted from the "slidingRP" metric developed by Nick Steinmetz's lab.

Algorithm
---------

**Step 1 — Spike Train Model**

The recorded spike train is modelled as a mixture of two independent processes:

- Base neuron spikes: N_b = (1 - C) * N_s spikes, which obey a true refractory period
  and therefore produce no close spike pairs among themselves.
- Contaminating spikes: N_c = C * N_s spikes, which are Poisson with no refractory
  period, where C is the contamination proportion (0 = clean, 1 = fully contaminated).

N_s is the total spike count and F_r is the total firing rate (Hz), so the recording
duration is approximately D = N_s / F_r.

**Step 2 — Observed Refractory Period Violations**

For each tested refractory period duration τ_r, the observed violation count V_o(τ_r)
is the cumulative sum of the one-sided autocorrelogram (ACG) from ref_acg_t_start up
to τ_r:

    V_o(τ_r) = Σ_{i=ref_acg_t_start}^{τ_r} n_ACG(i)

The ACG is one-sided (only positive lags), so each close spike pair is counted once.
ref_acg_t_start defaults to 0.25 ms to avoid bias from Kilosort's near-duplicate spike
removal (which suppresses spikes within ~0.25 ms of each other).  The effective tested
window is the adjusted refractory period τ_adj = τ_r − ref_acg_t_start.

**Step 3 — Expected Violations Under a Given Contamination Level**

For a one-sided ACG, two classes of spike pairs within [0, τ_adj] produce violations:

  - Contam–base pairs (either time ordering): 2 * N_c * N_b * τ_adj / D
  - Contam–contam pairs (unordered):         N_c * (N_c − 1) * τ_adj / D

Base–base pairs contribute zero violations because the base neuron has its own
refractory period.  Combining and substituting N_c = C*N_s, N_b = (1−C)*N_s:

    V_e = [2 * N_c * N_b + N_c * (N_c − 1)] * τ_adj / D
        ≈ C * (2 − C) * F_r * τ_adj * N_s

The factor (2 − C) reduces to ≈ 2 for small C and captures the contam–contam
self-interactions. This matches the formula in Llobet et al. (2022) and Steinmetz et al.

**Step 4 — Poisson Likelihood**

V_o is modelled as Poisson(V_e).  The likelihood of observing V_o or fewer violations
given contamination C is:

    P(X ≤ V_o | λ = V_e) = poisson.cdf(V_o, V_e)

If the true contamination were C, and we observe very few violations, this probability
will be small — the observations are unlikely under the hypothesis "contamination = C".

**Step 5 — Minimum Rejected Contamination**

For each cluster and each τ_r, find C_min: the smallest contamination for which the
Poisson likelihood falls below (1 − confidence), i.e. we can reject the hypothesis that
contamination is as large as C_min at the given confidence level.

Two equivalent methods are provided:

a) Binary search (compute_min_contam_props): bisects [0, max_contam_prop] to find C
   where poisson.cdf(V_o, V_e(C)) = 1 − confidence.

b) Analytical solution (compute_min_contam_props_analytical): uses the identity
       poisson.cdf(r, λ) = 1 − chi2.cdf(2λ, 2*(r+1))
   to solve for the critical Poisson rate in closed form:
       λ_crit = chi2.ppf(confidence, 2*(V_o+1)) / 2
   then inverts the quadratic  C * (2 − C) = k  (where k = λ_crit / (F_r * τ_adj * N_s))
   via the closed-form root:
       C = 1 − sqrt(1 − k)

Functions
---------
refractory_violation_likelihood     : Poisson CDF likelihood for a contamination level.
calc_rp_violations                  : Fast cumulative violation counts via k-th order ISIs.
compute_min_contam_props_analytical : Analytical (exact) solution for minimum rejected contamination.
compute_min_contam_props            : Binary search for minimum rejected contamination (legacy).
compute_rvl_tensor                  : Full likelihood tensor over a contam × RP grid.
plot_min_contam_prop                : Plot the minimum contamination curve vs. RP.
plot_rvl                            : Plot the full RVL likelihood heatmap.
"""
import numpy as np
from tqdm import tqdm
from scipy.stats import poisson, chi2
import matplotlib.pyplot as plt

from spike_utils.ccg import calc_local_firing_rate


def refractory_violation_likelihood(
        n_violations,
        contam_prop,
        refractory_period,
        firing_rate,
        n_spikes,
        ):
    '''
    Calculate the likelihood of an observed number of refractory period violations
    under a Poisson model that accounts for both contam-base and contam-contam
    spike pair interactions.

    The expected violation count is:

        V_e = C * (2 - C) * F_r * τ * N_s

    where C is the contamination proportion, F_r the firing rate, τ the refractory
    period, and N_s the spike count. The factor (2 - C) arises from counting both
    contam-base pairs (2*N_c*N_b) and contam-contam pairs (N_c*(N_c-1)).

    Parameters
    ----------
    n_violations : array_like
        The observed number of violations.
    contam_prop : array_like
        The contamination proportion to test (fraction of total spikes).
    refractory_period : array_like
        The refractory period in seconds.
    firing_rate : array_like
        The firing rate of the cluster in Hz.
    n_spikes : array_like
        The number of spikes in the cluster.

    Returns
    -------
    likelihood : float or array
        P(X ≤ n_violations | Poisson(V_e)), the CDF probability.
    '''
    expected_violations = contam_prop * (2 - contam_prop) * firing_rate * refractory_period * n_spikes
    likelihood = poisson.cdf(n_violations, expected_violations)
    return likelihood


def calc_rp_violations(spike_times, refractory_periods, ref_acg_t_start):
    """
    Compute cumulative refractory period violation counts using k-th order ISIs.

    Produces the same result as ``np.cumsum(calc_ccgs(st, np.r_[ref_acg_t_start,
    refractory_periods]).squeeze())``, but without the 3-D histogram overhead of
    calc_ccgs.  The ACG at lag τ equals the sum of all k-th order ISIs falling in
    [0, τ]:

        ACG(τ) = Σ_{k=1}^{∞} ISI_k(τ)

    For a 10 ms window, k rarely exceeds 2–3, so the outer loop terminates
    quickly.  Cumulative counts are then read off in O(n log n) via np.searchsorted
    rather than by binning.

    Parameters
    ----------
    spike_times : np.ndarray (n_spikes,)
        Sorted spike times in seconds.
    refractory_periods : np.ndarray (n_refractory_periods,)
        Refractory period upper bounds in seconds (monotonically increasing).
    ref_acg_t_start : float
        Lower bound of the violation window in seconds.

    Returns
    -------
    n_violations : np.ndarray (n_refractory_periods,) int
        Cumulative count of spike pairs with lag in [ref_acg_t_start, τ_r]
        for each τ_r.
    """
    max_tau = refractory_periods[-1]
    all_dts = []

    shift = 1
    while True:
        dts = spike_times[shift:] - spike_times[:-shift]
        valid_dts = dts[dts <= max_tau]
        if len(valid_dts) == 0:
            break
        all_dts.append(valid_dts)
        shift += 1

    if not all_dts:
        return np.zeros(len(refractory_periods), dtype=np.intp)

    all_dts = np.concatenate(all_dts)
    valid_dts = all_dts[all_dts >= ref_acg_t_start]
    valid_dts.sort()
    return np.searchsorted(valid_dts, refractory_periods, side='right')


def compute_min_contam_props_analytical(spike_times, spike_clusters=None, cids=None,
                       refractory_periods=np.exp(np.linspace(np.log(0.5e-3), np.log(10e-3), 100)),
                       max_contam_prop=1,
                       fr_est_dur=1,
                       confidence=0.95,
                       ref_acg_t_start=.25e-3,
                       progress=False,
                       device='cpu'):
    '''
    Compute the minimum contamination proportion that can be rejected using the exact
    analytical solution.

    Derivation
    ----------
    We want the smallest contamination C such that the Poisson likelihood of the observed
    violations V_o falls below (1 − confidence), i.e. find λ_crit satisfying:

        P(Poisson(λ_crit) ≤ V_o) = 1 − confidence                             ... (1)

    **Step 1 — Relate the Poisson CDF to the chi-squared CDF.**

    The Poisson CDF can be written in terms of the regularised upper incomplete gamma
    function Q(a, x) = Γ(a, x) / Γ(a):

        P(Poisson(λ) ≤ r) = Q(r + 1, λ)                                       ... (2)

    The chi-squared distribution with ν degrees of freedom is Gamma(ν/2, 2), so its CDF
    is the regularised *lower* incomplete gamma function:

        chi2.cdf(x, ν) = γ(ν/2, x/2) / Γ(ν/2)                                ... (3)

    Because the upper and lower incomplete gamma functions sum to the complete gamma,
    Q(a, x) = 1 − γ(a, x)/Γ(a).  Substituting a = r+1 and x = λ into (2) and comparing
    with (3) at ν = 2*(r+1) and x = 2λ:

        P(Poisson(λ) ≤ r) = Q(r+1, λ)
                           = 1 − γ(r+1, λ) / Γ(r+1)
                           = 1 − chi2.cdf(2λ, df=2*(r+1))                     ... (4)

    **Step 2 — Solve for λ_crit analytically.**

    Substituting (4) into (1):

        1 − chi2.cdf(2λ_crit, df=2*(V_o+1)) = 1 − confidence
        chi2.cdf(2λ_crit, df=2*(V_o+1))     = confidence
        2λ_crit = chi2.ppf(confidence, df=2*(V_o+1))
        λ_crit  = chi2.ppf(confidence, df=2*(V_o+1)) / 2                      ... (5)

    **Step 3 — Invert the expected-violations formula to recover C.**

    From the module docstring, V_e = C*(2−C)*F_r*τ_adj*N_s.  Setting V_e = λ_crit and
    letting k = λ_crit / (F_r * τ_adj * N_s):

        C*(2 − C) = k
        C^2 − 2C + k = 0
        C = 1 − sqrt(1 − k)       (smaller root, valid for C ∈ [0, 1])        ... (6)

    When k > 1 (i.e. λ_crit implies contamination above 100%) the result is capped at
    max_contam_prop via np.maximum(1 − k, 0) before taking the square root.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster IDs for each spike. If None, all spikes are assumed to be in the same cluster.
    cids : array-like (n_clusters,)
        Cluster IDs to test. Results returned in order of cids. If None, all clusters are tested.
    refractory_periods : array-like (n_refractory_periods,)
        Refractory periods to test in seconds.
    max_contam_prop : float
        Maximum contamination proportion to report.
    fr_est_dur : float
        Duration of the firing rate estimation window in seconds.
    confidence : float
        Confidence level for the test (e.g. 0.95 means 95% confidence).
    ref_acg_t_start : float
        Start time for the refractory period autocorrelogram in seconds.
        (necessary because Kilosort removes "duplicate" spikes within a .25 ms window)
    progress : bool
        Show a progress bar.
    device : str
        'cpu' (default) or 'cuda' for GPU-accelerated firing rate computation.

    Returns
    -------
    min_contam_props : array (n_clusters, n_refractory_periods)
        Minimum contamination proportion that can be rejected.
    firing_rates : array (n_clusters,)
        Firing rates for each cluster.
    '''
    spike_times = np.asarray(spike_times, dtype=np.float64).squeeze()

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int32).squeeze()
    assert spike_clusters.ndim == 1
    assert len(spike_times) == len(spike_clusters)

    if cids is not None:
        cids = np.asarray(cids, dtype=np.int32)
    else:
        cids = np.unique(spike_clusters)

    adj_ref_periods = refractory_periods - ref_acg_t_start

    firing_rates = np.zeros(len(cids))
    min_contam_props = np.ones((len(cids), len(refractory_periods))) * max_contam_prop
    for iC in tqdm(range(len(cids)), disable=not progress, desc="Calculating contamination (analytical)"):
        cid = cids[iC]
        st_clu = spike_times[spike_clusters == cid]
        n_spikes = len(st_clu)
        firing_rate = calc_local_firing_rate(st_clu, fr_est_dur)
        firing_rates[iC] = firing_rate
        n_violations = calc_rp_violations(st_clu, refractory_periods, ref_acg_t_start)

        # Analytical solution: invert C*(2-C) * F_r * tau * N_s = lambda_critical
        # lambda_critical = chi2.ppf(confidence, 2*(r+1)) / 2
        # C*(2-C) = k  =>  C^2 - 2C + k = 0  =>  C = 1 - sqrt(1 - k)
        lambda_critical = chi2.ppf(confidence, df=2 * (n_violations + 1)) / 2
        k = lambda_critical / (firing_rate * adj_ref_periods * n_spikes)
        contam = 1 - np.sqrt(np.maximum(1 - k, 0))
        min_contam_props[iC] = np.minimum(contam, max_contam_prop)

    return min_contam_props, firing_rates


def compute_min_contam_props(spike_times, spike_clusters=None, cids=None,
                       refractory_periods=np.exp(np.linspace(np.log(0.5e-3), np.log(10e-3), 100)),
                       max_contam_prop=1,
                       fr_est_dur=1,
                       confidence=0.95,
                       ref_acg_t_start=.25e-3,
                       progress=False,
                       tol=10e-4,
                       max_iter=100,
                       device='cpu'):
    '''
    Compute the minimum contamination proportion that can be rejected for each cluster
    via binary search over a Poisson model of refractory period violations.

    For most use cases, compute_min_contam_props_analytical is preferred (exact, faster).
    This function is provided as a reference/validation implementation.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster IDs for each spike. If None, all spikes are assumed to be in the same cluster.
    cids : array-like (n_clusters,)
        Cluster IDs to test. Results returned in order of cids. If None, all clusters are tested.
    refractory_periods : array-like (n_refractory_periods,)
        Refractory periods to test in seconds.
    max_contam_prop : float
        Maximum contamination proportion to test.
    fr_est_dur : float
        Duration of the firing rate estimation window in seconds.
    confidence : float
        Confidence level for the test (e.g. 0.95 means 95% confidence).
    ref_acg_t_start : float
        Start time for the refractory period autocorrelogram in seconds.
    progress : bool
        Show a progress bar.
    tol : float
        Convergence tolerance for the binary search.
    max_iter : int
        Maximum number of binary search iterations.
    device : str
        'cpu' (default) or 'cuda' for GPU-accelerated firing rate computation.

    Returns
    -------
    min_contam_props : array (n_clusters, n_refractory_periods)
        Minimum contamination proportion that can be rejected.
    firing_rates : array (n_clusters,)
        Firing rates for each cluster.
    '''
    spike_times = np.asarray(spike_times, dtype=np.float64).squeeze()

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int32).squeeze()
    assert spike_clusters.ndim == 1
    assert len(spike_times) == len(spike_clusters), "Spike times and spike clusters must have the same length."

    if cids is not None:
        cids = np.asarray(cids, dtype=np.int32)
        cids_check = np.unique(spike_clusters)
        assert np.all(np.isin(cids, cids_check)), "Some clusters are not in spike_clusters."
    else:
        cids = np.unique(spike_clusters)

    assert np.all(refractory_periods > 0), "Refractory periods must be positive."
    assert np.all(np.diff(refractory_periods) > 0), "Refractory periods must be monotonic."
    assert max_contam_prop > 0, "Contamination test proportions must be positive."

    firing_rates = np.zeros(len(cids))
    min_contam_props = np.ones((len(cids), len(refractory_periods))) * max_contam_prop
    for iC in tqdm(range(len(cids)), disable=not progress, desc="Calculating contamination"):
        cid = cids[iC]
        st_clu = spike_times[spike_clusters == cid]
        n_spikes = len(st_clu)
        firing_rate = calc_local_firing_rate(st_clu, fr_est_dur)
        firing_rates[iC] = firing_rate
        n_violations = calc_rp_violations(st_clu, refractory_periods, ref_acg_t_start)
        adj_ref_periods = refractory_periods - ref_acg_t_start

        # Vectorized binary search across all refractory periods simultaneously
        left = np.zeros(len(refractory_periods))
        right = np.full(len(refractory_periods), float(max_contam_prop))
        mid = np.empty(len(refractory_periods))
        for _ in range(max_iter):
            mid = (left + right) / 2
            likelihood = refractory_violation_likelihood(
                n_violations, mid, adj_ref_periods, firing_rate, n_spikes)
            too_high = likelihood < (1 - confidence)
            right = np.where(too_high, mid, right)
            left = np.where(too_high, left, mid)
            if np.max(right - left) < tol:
                break
        min_contam_props[iC] = mid

    return min_contam_props, firing_rates


def compute_rvl_tensor(spike_times, spike_clusters=None, cids=None,
                      refractory_periods=np.exp(np.linspace(np.log(0.5e-3), np.log(10e-3), 100)),
                      contamination_test_proportions=np.exp(np.linspace(np.log(5e-3), np.log(.35), 50)),
                      fr_est_dur=1,
                      ref_acg_t_start=.25e-3,
                      progress=False,
                      device='cpu'):
    '''
    Compute the likelihood of observing the number of refractory period violations
    or fewer for many clusters, refractory periods, and test contamination rates.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        The cluster ids for each spike. If None, all spikes are assumed to belong to a single cluster.
    cids : array-like (n_clusters,)
        Cluster IDs to include. If None, all unique clusters are used.
    refractory_periods : array-like (n_refrac,)
        The refractory periods to test, in seconds.
    contamination_test_proportions : array-like (n_contam,)
        The contamination rates to test, as a proportion of the firing rate.
    fr_est_dur : float
        The duration in seconds over which to estimate the firing rate.
    ref_acg_t_start : float
        The start time in seconds for the refractory period autocorrelogram.
    progress : bool
        Show a progress bar.
    device : str
        'cpu' (default) or 'cuda' for GPU-accelerated firing rate computation.

    Returns
    -------
    rvl_tensor : array
        A (n_clusters, n_contam, n_refrac) array with Poisson CDF likelihoods.
    '''
    spike_times = np.asarray(spike_times, dtype=np.float64).squeeze()

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int32).squeeze()
    assert spike_clusters.ndim == 1
    assert len(spike_times) == len(spike_clusters), "Spike times and spike clusters must have the same length."

    if cids is not None:
        cids = np.asarray(cids, dtype=np.int32)
        cids_check = np.unique(spike_clusters)
        assert np.all(np.isin(cids, cids_check)), "Some clusters are not in spike_clusters."
    else:
        cids = np.unique(spike_clusters)

    rvl_tensor = np.ones((len(cids), len(contamination_test_proportions), len(refractory_periods)))

    iter_range = range(len(cids))
    if progress:
        iter_range = tqdm(iter_range, desc="Calculating RVL tensor", position=0, leave=True)
    for iC in iter_range:
        cid = cids[iC]
        cluster_spikes = spike_times[spike_clusters == cid]
        n_spikes = len(cluster_spikes)
        firing_rate = calc_local_firing_rate(cluster_spikes, fr_est_dur)
        refractory_violations = calc_rp_violations(cluster_spikes, refractory_periods, ref_acg_t_start)

        rvl_tensor[iC] = refractory_violation_likelihood(
                            refractory_violations[None,:],
                            contamination_test_proportions[:,None],
                            refractory_periods[None,:] - ref_acg_t_start,
                            firing_rate,
                            n_spikes)

    return rvl_tensor


def plot_min_contam_prop(spike_times, min_contam_props, refractory_periods,
                         n_bins=50, max_contam_prop=1, acg_t_start=.25e-3, axs=None):
    '''
    Plot the minimum contamination proportion that can be rejected overlaid on the ISI distribution.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    min_contam_props : array (n_refractory_periods,)
        Minimum contamination rate that can be rejected.
    refractory_periods : array-like (n_refractory_periods,)
        Refractory periods tested, in seconds.
    n_bins : int
        Number of ISI histogram bins.
    max_contam_prop : float
        Y-axis upper limit for contamination proportion.
    acg_t_start : float
        ACG start time in seconds (for x-axis lower limit).
    axs : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes
    '''
    isis = np.diff(spike_times) * 1000
    max_refrac = refractory_periods.max() * 1000
    min_isi = acg_t_start * 1000
    min_prop = min_contam_props.min()

    if axs is None:
        fig, axs = plt.subplots(1, 1)
    else:
        fig = axs.get_figure()
    bins = np.linspace(min_isi, max_refrac, n_bins)
    axs.hist(isis, bins=bins, edgecolor='black', color='black', alpha=0.6)
    axs.set_xlim([min_isi, max_refrac])
    axs.set_ylabel('ISI count (spikes)')
    axs.set_xlabel('ISI / Refractory Period (ms)')
    axs2 = axs.twinx()
    axs2.plot(refractory_periods*1000, min_contam_props, color='red', linewidth=3.5)
    axs2.axhline(min_prop, color='red', linestyle='--', linewidth=2)
    yticks = np.concatenate([np.linspace(0, max_contam_prop, 6), [min_prop]])
    axs2.set_ylim([0, max_contam_prop])
    axs2.set_yticks(yticks)
    axs2.set_yticklabels(['0', '', '', '', '', '1', f'{min_prop:.4g}'])
    axs2.tick_params(axis='y', colors='red')
    axs2.set_ylabel('Minimum Rejected Contamination Proportion', color='red')

    return fig, axs


def plot_rvl(cluster_spikes, likelihoods, refractory_periods, contamination_test_proportions, likelihood_threshold=0.05):
    '''
    Plot the full RVL likelihood heatmap with ISI distribution and min contamination curve.

    Parameters
    ----------
    cluster_spikes : array-like (n_spikes,)
        Spike times for a single cluster.
    likelihoods : array (n_contam, n_refrac)
        Likelihood tensor for a single cluster.
    refractory_periods : array-like (n_refrac,)
        Refractory periods tested, in seconds.
    contamination_test_proportions : array-like (n_contam,)
        Contamination proportions tested.
    likelihood_threshold : float
        Threshold for rejecting contamination hypothesis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : array of matplotlib.axes.Axes (3,)
    '''
    min_refrac, max_refrac = refractory_periods.min(), refractory_periods.max()
    min_contam, max_contam = contamination_test_proportions.min(), contamination_test_proportions.max()

    isis = np.diff(cluster_spikes)

    min_likelihood_per_contam = np.min(likelihoods, axis=1)
    min_likelihood_per_contam[min_likelihood_per_contam < likelihood_threshold] = np.inf
    lowest_contam_idx = np.argmin(min_likelihood_per_contam)
    lowest_contam = contamination_test_proportions[lowest_contam_idx]
    lowest_contam_likelihood = likelihoods[lowest_contam_idx]

    fig, axs = plt.subplots(3, 1, figsize=(5, 12), height_ratios=[1, 1.5, 1])
    axs[0].hist(isis * 1000, bins=np.arange(0, max_refrac*1000, .33))
    axs[0].set_title('ISI distribution')
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('ISI (ms)')
    axs[0].set_xlim([0, max_refrac*1000])

    extent = [min_refrac*1000, max_refrac*1000, min_contam, max_contam]
    from matplotlib.image import NonUniformImage
    im = NonUniformImage(axs[1], extent=extent, interpolation='nearest', cmap='viridis')
    im.set_data(refractory_periods*1000, contamination_test_proportions, likelihoods)
    im.set_clim(0, 1)
    axs[1].add_image(im)
    axs[1].axhline(lowest_contam, color='red', linestyle='--')
    axs[1].set_xlim(extent[:2])
    axs[1].set_ylim(extent[2:])
    fig.colorbar(im, ax=axs[1], orientation='horizontal', label='Likelihood')
    axs[1].set_title('Likelihood of observed refractory period violations')
    axs[1].set_xlabel('Refractory period (ms)')
    axs[1].set_ylabel('Contamination rate')

    axs[2].semilogy(refractory_periods*1000, lowest_contam_likelihood)
    axs[2].axhline(likelihood_threshold, color='red', linestyle='--')
    axs[2].set_title(f'Highest contamination rate more than {likelihood_threshold*100:.1g}% likely: {lowest_contam*100:.3g}%')
    axs[2].set_xlabel('Refractory period (ms)')
    axs[2].set_ylabel('Likelihood')
    axs[2].set_xlim([min_refrac*1000, max_refrac*1000])

    plt.tight_layout()
    return fig, axs
