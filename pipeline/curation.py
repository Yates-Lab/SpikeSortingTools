#%%
#Curation with the SortingAnalyzer, to clean up the sorting results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tqdm import tqdm
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import remove_duplicated_spikes
from spikeinterface.curation import remove_redundant_units
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.curation import auto_merge_units
from spikeinterface.curation import apply_curation
from spikeinterface.curation import find_redundant_units
from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift, get_template_amplitudes
from spikeinterface.postprocessing import align_sorting
from spikeinterface.exporters.to_phy import export_to_phy
from spikeinterface.extractors import read_phy
from spikeinterface.sorters import KilosortSorter


def automerge(analyzer):
    #Biggest issue is temporal shifts temporal_splits

    # some extensions are required
    # analyzer.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
    # analyzer.compute("unit_locations", method="monopolar_triangulation")

    # presence_distance_thresh = [100]
    # presets = ["temporal_splits"] * len(presence_distance_thresh)
    # steps_params = [
    #     {"presence_distance": {"presence_distance_thresh": i}}
    #     for i in presence_distance_thresh
    # ]
    

    # # template_diff_thresh = [0.05, 0.15, 0.25]
    # # presets += ["x_contaminations"] * len(template_diff_thresh)
    # # steps_params += [
    # #     {"template_similarity": {"template_diff_thresh": i}}
    # #     for i in template_diff_thresh
    # # ]

    # compute_merge_args={
    #     "preset": presets,
    #     "steps_params": steps_params,
    #     "recursive": True
    # }
    compute_merge_args={
        "preset": "temporal_splits"
    }#    "recursive": True
    #} #     "presence_distance_thresh": [100],
    analyzer_merged = auto_merge_units(
        sorting_analyzer=analyzer,
        compute_merge_kwargs=compute_merge_args
    )

    # merge_unit_groups = get_potential_auto_merge(
    # analyzer=analyzer,
    # preset="similarity_correlograms",
    # resolve_graph=True
    # )

    # # here we apply the merges
    # analyzer_merged = analyzer.merge_units(merge_unit_groups=merge_unit_groups)
    return analyzer_merged

def run_cur(seg, ks4_sorter, cache_dir, recalc=False):
    '''
    Run the curation pipeline on the given sorted data.
    
    Parameters
    ----------
    seg: spikeinterface recording segment
        The recording segment which was sorted. Used to extract waveforms and other data.
    results: KilosortResults
        The results of the kilosort4 sorting.
    
    Returns
    -------
    cur_results: dict
        The results of the quality control pipeline
    '''

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'cur_todo_phy.npz'
    
    if npz_path.exists() and not recalc:
        curation_todo = np.load(npz_path)
        return curation_todo
    

    analyzer = create_sorting_analyzer(sorting=ks4_sorter, recording=seg)
    # # some extensions are required
    analyzer.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
    analyzer.compute("unit_locations", method="monopolar_triangulation")

    merge_unit_groups = compute_merge_unit_groups(analyzer,preset="temporal_splits", presence_distance=100)
    

    #redundant, bad units
    remove_unit_ids = []

    #copying from remove_redundant_units, but without applying the removal (yet)
    remove_strategy = "minimum_shift"
    peak_sign="neg"

    unit_peak_shifts = get_template_extremum_channel_peak_shift(analyzer)
    sorting_aligned = align_sorting(sorting=ks4_sorter, unit_peak_shifts=unit_peak_shifts)
    redundant_unit_pairs= find_redundant_units(sorting=sorting_aligned, delta_time = 0.4, agreement_threshold=0.2, duplicate_threshold=0.8)

    if remove_strategy in ("minimum_shift", "highest_amplitude"):
        # this is the values at spike index !
        peak_values = get_template_amplitudes(analyzer, peak_sign=peak_sign, mode="at_index")
        peak_values = {unit_id: np.max(np.abs(values)) for unit_id, values in peak_values.items()}

    if remove_strategy == "minimum_shift":
        #assert align, "remove_strategy with minimum_shift needs align=True"
        for u1, u2 in redundant_unit_pairs:
            if np.abs(unit_peak_shifts[u1]) > np.abs(unit_peak_shifts[u2]):
                remove_unit_ids.append(u1)
            elif np.abs(unit_peak_shifts[u1]) < np.abs(unit_peak_shifts[u2]):
                remove_unit_ids.append(u2)
            else:
                # equal shift use peak values
                if np.abs(peak_values[u1]) < np.abs(peak_values[u2]):
                    remove_unit_ids.append(u1)
                else:
                    remove_unit_ids.append(u2)
    
    curation_todo = {
        "merge_unit_groups": merge_unit_groups,
        "removed_units":remove_unit_ids,
    }

    export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy')
    np.savez(npz_path, **curation_todo)
    #ideally save to cluster_info.tsv and cluster_group.tsv
    #export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy')
    return curation_todo


    # # Prepare curation dictionary
    # label_definitions={
    #     "quality": {
    #         "label_options": [
    #             "good",
    #             "noise",
    #             "mua",
    #             "artifact"
    #         ],
    #         "exclusive": "true"
    #     },
    #     "putative_type": {
    #         "label_options": [
    #             "excitatory",
    #             "inhibitory",
    #             "pyramidal",
    #             "mitral"
    #         ],
    #         "exclusive": "false"
    #     }
    # }
    
    # ks_labels = ks4_sorter.get_property('KSLabel')
    # ks_ids=ks4_sorter.unit_ids

    # curation_dict = {
    #     "format_version": "1",
    #     "unit_ids": ks_ids,
    #     "label_definitions": label_definitions,
    #     "manual_labels": ks_labels, #need to add unit_ids to this, or change curation_dict behavior
    #     "merge_unit_groups": merge_unit_groups,
    #     "removed_units":remove_unit_ids,
    # }
    
    # cur_sorter=apply_curation(ks4_sorter, curation_dict=curation_dict)
    # return cur_sorter

    # # merge units with similar templates and correlograms
    # analyzer.compute("waveforms")
    # analyzer_merged=automerge(analyzer)

    # #these shouldn't need to be recomputed, they should inherit the analyzer_merged
    # # analyzer_merged.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
    # # analyzer_merged.compute("unit_locations", method="monopolar_triangulation")
    
    # # remove redundant units from SortingAnalyzer object
    # # note this returns a cleaned sorting
    # clean_sorting = remove_redundant_units(
    #     analyzer_merged,
    #     duplicate_threshold=0.9,
    #     remove_strategy="minimum_shift"
    # )
    # # in order to have a SortingAnalyer with only the non-redundant units one must
    # # select the designed units remembering to give format and folder if one wants
    # # a persistent SortingAnalyzer.
    # clean_sorting_analyzer = analyzer_merged.select_units(clean_sorting.unit_ids)

    # clean_sorting_analyzer.save_as(format="binary_folder", folder = cache_dir / 'clean_sorting_analyzer')
    # #apply_curation(ks4_sorter, curation_dict=)
    # export_to_phy(clean_sorting_analyzer, cache_dir / 'clean_sorting_analyzer_phy')
    # cur_sorting = read_phy(cache_dir / 'clean_sorting_analyzer') # Pull from output directory
    # return cur_sorting

    # cur_results = {}

    # spike_samples = clean_sorting_analyzer.spike_times
 

    #return clean_sorting_analyzer

def load_cur(cache_dir):
    '''
    Load the quality control results from a given directory.
    
    Parameters
    ----------
    cache_dir: str or Path
        The directory to load the quality control results from.
    
    Returns
    -------
    cur_results: dict
        The quality control results
    '''
    cur_results=np.load(cache_dir)

    return cur_results


#%%
