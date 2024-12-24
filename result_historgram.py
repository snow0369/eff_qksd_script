import json
import os
import pickle
from copy import deepcopy
from time import time

import numpy as np
from matplotlib import pyplot as plt

from plot_tool import plot_histogram
from ofex.measurement import sorted_insertion
from ofex_algorithms.qksd import qksd_shot_allocation, sample_qksd
from ofex_algorithms.qksd.qksd_utils import trunc_eigh
from base_utils import prepare_main, approx_sampling_noise_norm_s, path_probbuf
from ics_interface import ics_interface
from result_base_setting import default_transform, default_n_trotter, default_shift_opt_lvl

DIR_RESULT_HISTOGRAM = "./histogram/"


def run_qksd(data, num_workers, tot_shots, n_batches):
    mol_name = data['mol_name']

    if not os.path.exists(DIR_RESULT_HISTOGRAM):
        os.makedirs(DIR_RESULT_HISTOGRAM)
    raw_data_path = os.path.join(DIR_RESULT_HISTOGRAM, f"{mol_name}_qksd_matrices.pkl")
    if os.path.exists(raw_data_path):
        with open(raw_data_path, "rb") as f:
            raw_data = pickle.load(f)
            result_h_samp = {method: np.array(h_samp) for method, h_samp in raw_data["result_h_samp"].items()}
            result_s_samp = {method: np.array(s_samp) for method, s_samp in raw_data["result_s_samp"].items()}
            exact_h = np.array(raw_data["exact_h"])
            exact_s = np.array(raw_data["exact_s"])
            return result_h_samp, result_s_samp, exact_h, exact_s

    result_h_samp = {"SI_FH": None,
                     "SI_LCU": None,
                     "SHIFT_SI_FH": None,
                     "SHIFT_SI_LCU": None,
                     "SHIFT_ICS_CISD_FH": None,
                     "SHIFT_ICS_CISD_LCU": None,}
    result_s_samp = deepcopy(result_h_samp)
    exact_shift_h, exact_s, shfit_const = data["h_mat"], data["s_mat"], data["shift_const"]
    exact_h = exact_shift_h + exact_s * shfit_const

    ref, pham, cisd_state, n_qksd, propagator = (data['ref'], data['pham'], data["cisd_state"],
                                                 data['n_qksd'], data['propagator'])
    _, shift_pham, shift_const = data["shift_dict"][default_shift_opt_lvl]

    # Perform ICS with shifting
    frag_shot_list = ics_interface(data, cisd_state, "CISD", num_workers,
                                   anticommute=True, shift_opt_lvl=default_shift_opt_lvl)
    grp_ham_cisd_lcu = [frag for frag, shot, _, _ in frag_shot_list]
    shot_cisd_lcu = [shot for frag, shot, _, _ in frag_shot_list]

    frag_shot_list = ics_interface(data, cisd_state, "CISD", num_workers,
                                   anticommute=False, shift_opt_lvl=default_shift_opt_lvl)
    grp_ham_cisd_fh = [frag for frag, shot, _, _ in frag_shot_list]
    shot_cisd_fh = [shot for frag, shot, _, _ in frag_shot_list]

    # Sorted insertion
    grp_ham_si = {"SI_FH": sorted_insertion(pham, False),
                  "SI_LCU": sorted_insertion(pham, True),
                  "SHIFT_SI_FH": sorted_insertion(shift_pham, False),
                  "SHIFT_SI_LCU": sorted_insertion(shift_pham, True),}
    frag_shot_si = {"SI_FH": None, "SI_LCU": None, "SHIFT_SI_FH": None, "SHIFT_SI_LCU": None}
    for method, grp_ham in grp_ham_si.items():
        frag_shot = np.zeros((2, len(grp_ham)))
        frag_shot[0, :] = np.array([frag.induced_norm(order=2) for frag in grp_ham])
        frag_shot[1, :] = np.array([frag.induced_norm(order=2) for frag in grp_ham])
        frag_shot /= np.sum(frag_shot)
        frag_shot_si[method] = frag_shot

    for method in result_h_samp.keys():
        print(method, end=' ')
        t = time()
        meas_opt_name = '_'.join(method.split("_")[:-1])
        if "FH" in method:
            anticommute = False
        elif "LCU" in method:
            anticommute = True
        else:
            raise AssertionError
        meas_type = "FH" if not anticommute else "LCU"

        # For SI cases:
        if method in grp_ham_si:
            grp_ham, frag_shot = grp_ham_si[method], frag_shot_si[method]
        # For ICS+SHIFT method
        elif "CISD" in method:
            grp_ham = grp_ham_cisd_lcu if anticommute else grp_ham_cisd_fh
            frag_shot = shot_cisd_lcu if anticommute else shot_cisd_fh
        else:
            raise AssertionError

        # Perform QKSD sampling
        shot_alloc = qksd_shot_allocation(tot_shots, grp_ham, n_qksd, meas_type, is_toeplitz=True,
                                          frag_shot_alloc=np.array(frag_shot))

        samp_h, samp_s = sample_qksd(grp_ham, propagator, ref, n_qksd, is_toeplitz=True,
                                     meas_type=meas_type, shot_list=shot_alloc, n_batch=n_batches,
                                     sample_buf_dir=path_probbuf(mol_name, default_transform, anticommute, meas_opt_name))

        if "SHIFT" in method:
            samp_h += samp_s * shift_const
        result_h_samp[method] = samp_h.tolist()
        result_s_samp[method] = samp_s.tolist()
        print(f"time = {time()-t:.3f} sec")

    with open(raw_data_path, "wb") as f:
        raw_data = {"result_h_samp": result_h_samp, "result_s_samp": result_s_samp,
                    "exact_h": exact_h.tolist(), "exact_s": exact_s.tolist()}
        pickle.dump(raw_data, f)

    return ({method: np.array(h_samp) for method, h_samp in result_h_samp.items()},
            {method: np.array(s_samp) for method, s_samp in result_s_samp.items()},
            np.array(exact_h), np.array(exact_s))

def qksd_eig_result(data, result_h_samp, result_s_samp, exact_h, exact_s, tot_shots, n_batches, save_json):
    mol_name = data['mol_name']

    eig_data_path = os.path.join(DIR_RESULT_HISTOGRAM, f"{mol_name}_eig_data.pkl")
    if os.path.exists(eig_data_path):
        with open(eig_data_path, "rb") as f:
            result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd = pickle.load(f)
        if save_json:
            s = json.dumps({"error": result_err, "ideal_qksd": ideal_qksd_gnd, "ideal_qksd_trunc": ideal_qksd_trunc_gnd}, )
            with open(os.path.join(DIR_RESULT_HISTOGRAM, f"{mol_name}_eig_data.json"), "w") as f:
                f.write(s)
        return result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd

    n_qksd = data['n_qksd']

    # FCI value
    gnd_fci = data['gnd_energy']

    # Exact QKSD without truncation
    val, vec = trunc_eigh(exact_h, exact_s, epsilon=1e-12)
    ideal_qksd_gnd = np.min(val) - gnd_fci
    assert np.isclose(ideal_qksd_gnd, 0.0, atol=1e-3), ideal_qksd_gnd

    # Noisy QKSD
    n_prime = {"FH": list(), "LCU": list()}

    result_err = dict()
    for method in result_h_samp.keys():
        result_err[method] = list()
        anticommute = ("LCU" in method)
        meas_type = "FH" if not anticommute else "LCU"

        eps = approx_sampling_noise_norm_s(tot_shots, n_qksd, is_lcu=anticommute) # Trunc. param.
        h_samp, s_samp = np.array(result_h_samp[method]), np.array(result_s_samp[method])
        for b in range(n_batches):
            h_samp_b, s_samp_b = h_samp[b], s_samp[b]
            val, vec = trunc_eigh(h_samp_b, s_samp_b, epsilon=eps)
            n_prime[meas_type].append(len(val)) # n_prime doesn't depends on meas. opt. for H
            result_err[method].append(np.min(val) - gnd_fci)

    # Exact QKSD with truncation
    ideal_qksd_trunc_gnd = dict()
    for meas_type in ["FH", "LCU"]:
        n = int(np.round(np.average(n_prime[meas_type])))
        val, vec = trunc_eigh(exact_h, exact_s, n_lambda=n)
        ideal_qksd_trunc_gnd[meas_type] = np.min(val) - gnd_fci

    with open(eig_data_path, "wb") as f:
        pickle.dump((result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd), f)

    if save_json:
        s = json.dumps({"error": result_err, "ideal_qksd": ideal_qksd_gnd, "ideal_qksd_trunc": ideal_qksd_trunc_gnd}, )
        with open(os.path.join(DIR_RESULT_HISTOGRAM, f"{mol_name}_eig_data.json"), "w") as f:
            f.write(s)

    return result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd


def plot_qksd_histogram(data, result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd, ):
    mol_name = data['mol_name']
    method_list = ["SI", "SHIFT_SI", "SHIFT_ICS_CISD"]
    method_list_fh = [method+"_FH" for method in method_list]
    method_list_lcu = [method+"_LCU" for method in method_list]

    label_list = [r"$\tilde{E}^{(n\rightarrow n')}$(SI)",
                  r"$\tilde{E}^{(n\rightarrow n')}$(SHIFT, SI)",
                  r"$\tilde{E}^{(n\rightarrow n')}$(SHIFT, ICS)"]

    plt.figure(figsize=(8,6))
    fig, axes = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.4)

    # * 1000 for rescaling Ha to mHa
    plot_histogram(axes[0],
                   [np.array(result_err[method]) * 1000 for method in method_list_fh],
                   label_list,
                   v_bars=[ideal_qksd_gnd * 1000, ideal_qksd_trunc_gnd["FH"] * 1000],
                   v_labels=[r"$E^{(n)}$" r"$E^{(n\rightarrow n')}$"],
                   n_bins=250,
                   bound_x = (-2.0, 2.0))
    plot_histogram(axes[1], [np.array(result_err[method]) * 1000 for method in method_list_lcu],
                   label_list,
                   v_bars=[ideal_qksd_gnd * 1000, ideal_qksd_trunc_gnd["LCU"] * 1000],
                   v_labels=[r"$E^{(n)}$" r"$E^{(n\rightarrow n')}$"],
                   n_bins=250,
                   bound_x = (-2.0, 2.0))

    axes[0].legend()
    axes[0].set_ylim(0, 0.04)
    axes[1].set_ylim(0, 0.04)

    axes[1].set_xlabel(r"$\tilde{E}^{(n\rightarrow n')}$(mHa)")
    axes[0].set_title("FH (Extended Swap Test)")
    axes[1].set_title("LCU (Hadamard Test)")

    plt.savefig(os.path.join(DIR_RESULT_HISTOGRAM, f"{mol_name}_histogram.png"))
    plt.close()

def main():
    num_workers = 8
    tot_shots = 1e8
    n_batches = 10000
    n_qksd = 10

    data = prepare_main("H2O", default_transform, default_n_trotter, n_qksd,
                             default_shift_opt_lvl, gap_estimation="overlap")
    print("Run QKSD Started")
    # Takes long time
    result_h_samp, result_s_samp, exact_h, exact_s = run_qksd(data, num_workers, tot_shots, n_batches)

    print("Eig Analysis Started")
    result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd = qksd_eig_result(data, result_h_samp, result_s_samp,
                                                                       exact_h, exact_s,
                                                                       tot_shots, n_batches,
                                                                       save_json=True)

    print("Plotting Started")
    plot_qksd_histogram(data, result_err, ideal_qksd_gnd, ideal_qksd_trunc_gnd)

if __name__ == "__main__":
    main()
