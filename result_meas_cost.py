"""Calculate true / empirical variances of fragments with
 - sorted_insertion
 - ICS(with true QKSD states)
 - ICS(with CISD)
 - shift
 - shift + ICS(with CISD)
"""
import csv
import os
import pickle
from itertools import product

import numpy as np

from ofex.linalg.sparse_tools import apply_operator
from ofex.measurement import fragment_variance, sorted_insertion
from ofex.measurement.iterative_coefficient_splitting import init_ics
from ofex.state.state_tools import to_dense
from ofex_algorithms.qksd import qksd_shot_allocation, sample_qksd
from base_utils import prepare_main, path_probbuf
from ics_interface import ics_interface, path_cov
from result_base_setting import default_transform, default_n_trotter, default_n_qksd, default_shift_opt_lvl, \
    default_mol_list


def true_variance(mol_name, num_workers):
    data = prepare_main(mol_name, default_transform, default_n_trotter, default_n_qksd[mol_name],
                        default_shift_opt_lvl,
                        gap_estimation="energy")
    pham, n_qksd, cisd_state, ref1, propagator = data['pham'], data['n_qksd'], data['cisd_state'], data['ref'], data['propagator']
    _, shift_pham, _ = data["shift_dict"][default_shift_opt_lvl]

    result = {
        "SI_FH": list(),
        "SI_LCU": list(),
        "SHIFT_SI_FH": list(),
        "SHIFT_SI_LCU": list(),
        "ICS_TRUE_FH": list(),
        "ICS_TRUE_LCU": list(),
        "ICS_CISD_FH": list(),
        "ICS_CISD_LCU": list(),
        "SHIFT_ICS_CISD_FH": list(),
        "SHIFT_ICS_CISD_LCU": list(),
    }

    true_state = to_dense(ref1)

    ics_cisd_list = dict()

    # Perform ICS with CISD
    for anticommute, shift_opt_lvl in product([True, False], [None, default_shift_opt_lvl]):
        # First, calculate CISD ICS efficiently with estimated CISD phase
        ics_cisd_list[anticommute, shift_opt_lvl] = ics_interface(data, cisd_state, "CISD", num_workers,
                                                                  anticommute, shift_opt_lvl)

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

    for k in range(n_qksd):
        for anticommute, shift_opt_lvl in product([True, False], [None, default_shift_opt_lvl]):
            print(f"anticommute = {anticommute}, shift_opt_lvl = {shift_opt_lvl}, k = {k}")
            frag_type = "FH" if not anticommute else "LCU"

            # For Non-shifting case, compute true ICS case
            if shift_opt_lvl is None:
                grp_ham_true, frag_shots_ics, _, _ = ics_interface(data, true_state, f"TRUE_{k}",
                                                                   num_workers, anticommute, shift_opt_lvl)
                name_result = f"ICS_TRUE_{frag_type}"
                result[name_result].append(fragment_variance(grp_ham_true, ref1, true_state, frag_shots_ics,
                                                             anticommute=anticommute))

            true_cov_buf_dir = path_cov(mol_name, default_transform, anticommute, shift_opt_lvl=None, ref2_name=f"TRUE_{k}")
            assert os.path.exists(true_cov_buf_dir)
            _, true_cov_dict = init_ics(pham, ref1, true_state, num_workers=num_workers, anticommute=anticommute,
                                        cov_buf_dir=true_cov_buf_dir, phase_list=None)

            # Calculate CISD ICS frag variance
            grp_ham_cisd, frag_shot_cisd, _, _ = ics_cisd_list[anticommute, shift_opt_lvl][k]
            name_result = f"SHIFT_ICS_CISD_{frag_type}" if shift_opt_lvl is not None else f"ICS_CISD_{frag_type}"
            result[name_result].append(
                fragment_variance(grp_ham_cisd, ref1, true_state, frag_shot_cisd,
                                  true_cov_dict=true_cov_dict,
                                  anticommute=anticommute))

            # Calculate SI frag varaince
            si_method = f"SI_{frag_type}" if shift_opt_lvl is None else f"SHIFT_SI_{frag_type}"
            result[si_method].append(
                fragment_variance(grp_ham_si[si_method], ref1, true_state, frag_shot_si[si_method],
                                  true_cov_dict=true_cov_dict,
                                  anticommute=anticommute))

        if k != n_qksd - 1:
            true_state = apply_operator(propagator, true_state)

    return result


def empirical_variance(mol_name, num_workers, tot_shots, n_batch):
    data = prepare_main(mol_name, default_transform, default_n_trotter, default_n_qksd[mol_name], default_shift_opt_lvl,
                        gap_estimation="energy")
    ideal_h_shifted = data["h_mat"]
    ideal_h_unshifted = data["h_mat"] + data["s_mat"] * data["shift_const"]

    pham, propagator, ref, n_qksd = data["pham"], data["propagator"], data['ref'], data["n_qksd"]
    _, shift_pham, _ = data["shift_dict"][default_shift_opt_lvl]

    tot_shots_qksd = tot_shots * (n_qksd - 1)

    result = dict()
    result_deviation = dict()
    raw_data = dict()

    for meas_opt_name, anticommute in product(["SI", "SHIFT_SI", "ICS_TRUE", "ICS_CISD", "SHIFT_ICS_CISD"], [True, False]):
        meas_type = "FH" if not anticommute else "LCU"
        print(meas_opt_name, meas_type)
        if "SHIFT" in meas_opt_name:
            shift_opt_lvl = default_shift_opt_lvl
            ideal_h = ideal_h_shifted[0, :]
        else:
            shift_opt_lvl = None
            ideal_h = ideal_h_unshifted[0, :]
        if "TRUE" in meas_opt_name:
            frag_shot_list = [ics_interface(data, None, f"TRUE_{k}", num_workers, anticommute, shift_opt_lvl)
                              for k in range(n_qksd)]
            ham_frag = [frag for frag, shot, _, _ in frag_shot_list]
            shots = np.array([shot for frag, shot, _, _ in frag_shot_list])
        elif "CISD" in meas_opt_name:
            frag_shot_list = ics_interface(data, None, f"CISD", num_workers, anticommute, shift_opt_lvl)
            ham_frag = [frag for frag, shot, _, _ in frag_shot_list]
            shots = np.array([shot for frag, shot, _, _ in frag_shot_list])
        elif "SI" in meas_opt_name:
            if "SHIFT" in meas_opt_name:
                ham_frag = sorted_insertion(shift_pham, anticommute)
            else:
                ham_frag = sorted_insertion(pham, anticommute)
            shots = np.zeros((2, len(ham_frag)))
            shots[0, :] = np.array([frag.induced_norm(order=2) for frag in ham_frag])
            shots[1, :] = np.array([frag.induced_norm(order=2) for frag in ham_frag])
            shots /= np.sum(shots)
        else:
            raise AssertionError

        shot_alloc = qksd_shot_allocation(tot_shots_qksd if meas_type=="FH" else tot_shots_qksd * 2,
                                          ham_frag, n_qksd, meas_type, is_toeplitz=True,
                                          frag_shot_alloc=shots)

        samp_h, _ = sample_qksd(ham_frag, propagator, ref, n_qksd, is_toeplitz=True,
                                meas_type=meas_type, shot_list=shot_alloc, n_batch=n_batch,
                                sample_buf_dir=path_probbuf(mol_name, default_transform, anticommute, meas_opt_name))
        err = np.zeros((n_batch, n_qksd), dtype=complex)
        for i in range(n_batch):
            err[i, :] = np.abs(samp_h[i, 0, :] - ideal_h) ** 2 * tot_shots
        result[f"{meas_opt_name}_{meas_type}"] = list(np.average(err, axis=0))
        result_deviation[f"{meas_opt_name}_{meas_type}"] = list(np.std(err, axis=0))
        raw_data[f"{meas_opt_name}_{meas_type}"] = ideal_h.tolist()
    return result, result_deviation, raw_data

DIR_RESULT_MEAS_COST = "./measurement_cost/"
DIR_RAW_DATA = "./measurement_cost/raw_data/"
def result_generator(num_workers, tot_shots, n_batch):
    if not os.path.exists(DIR_RESULT_MEAS_COST):
        os.makedirs(DIR_RESULT_MEAS_COST)
    if not os.path.exists(DIR_RAW_DATA):
        os.makedirs(DIR_RAW_DATA)

    methods = list()
    result = dict()
    for mol_name in default_mol_list:
        path_true = os.path.join(DIR_RESULT_MEAS_COST, f"{mol_name}_TRUEVAR.pkl")
        path_empi = os.path.join(DIR_RESULT_MEAS_COST, f"{mol_name}_EMPIVAR.pkl")
        path_empi_raw = os.path.join(DIR_RAW_DATA, f"{mol_name}_EMPIRAW.pkl")
        if not os.path.exists(path_true):
            result_true = true_variance(mol_name, num_workers)
            with open(path_true, "wb") as f:
                pickle.dump(result_true, f)
        else:
            with open(path_true, "rb") as f:
                result_true = pickle.load(f)
        if not (os.path.exists(path_empi) and os.path.exists(path_empi_raw)):
            result_empi, dev_empi, raw_data = empirical_variance(mol_name, num_workers, tot_shots, n_batch)
            with open(path_empi, "wb") as f:
                pickle.dump((result_empi, dev_empi), f)
            with open(path_empi_raw, "wb") as f:
                pickle.dump(raw_data, f)
        else:
            with open(path_empi, "rb") as f:
                result_empi, dev_empi = pickle.load(f)
        print(f"========================= {mol_name} =====================")
        result[mol_name] = dict()
        assert set(result_true.keys()) == set(result_empi.keys())
        for k in result_true.keys():
            print(f"\t{k}")
            print(f"\t\t{result_true[k]}")
            print(f"\t\t{result_empi[k]}")
            print(f"\t\t{dev_empi[k]}")
            result[mol_name][k+"_TRUE"] = np.average(result_true[k][1:])
            result[mol_name][k+"_EMPI"] = np.average(result_empi[k][1:])
        print(f"============================================================")

        methods = sorted(result[mol_name].keys())

    with open(os.path.join(DIR_RESULT_MEAS_COST, "total_result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_name"] + methods)
        for mol_name in default_mol_list:
            writer.writerow([mol_name] + [result[mol_name][method] for method in methods])

if __name__ == '__main__':
    result_generator(num_workers=8, tot_shots=1e8, n_batch=1000)
