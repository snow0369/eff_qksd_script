import os
import pickle
from datetime import datetime
from time import time
from typing import List, Optional, Any

import numpy as np
from openfermion import QubitOperator
from tqdm import tqdm

from ofex.measurement.iterative_coefficient_splitting import init_ics, run_ics
from base_utils import rec_path

DIR_ICS = "./ics_data/"
DIR_COV = "./cov_data/"


def path_ics(mol_name, transform, anticommute, shift_opt_lvl, ref2_name):
    grp_type = "FH" if not anticommute else "LCU"
    shift_type = f"shifted-{shift_opt_lvl}" if shift_opt_lvl is not None else "unshifted"
    return rec_path((DIR_ICS, f"{mol_name}_{transform}", shift_type), "ics.pkl", f"{grp_type}_{ref2_name}")


def path_cov(mol_name, transform, anticommute, shift_opt_lvl, ref2_name):
    grp_type = "FH" if not anticommute else "LCU"
    shift_type = f"shifted-{shift_opt_lvl}" if shift_opt_lvl is not None else "unshifted"
    return rec_path((DIR_COV, f"{mol_name}_{transform}", shift_type), "cov.pkl", f"{grp_type}_{ref2_name}")


def pickle_ics(frag_list, shot_alloc_arr, final_variance, cvec):
    pkl_frag_list = [frag.terms for frag in frag_list]
    return pkl_frag_list, shot_alloc_arr.tolist(), final_variance, cvec.tolist()


def unpickle_ics(pkl_frag_list, pkl_shot_alloc_arr, final_variance, pkl_cvec):
    frag_list = list()
    for pkl_frag in pkl_frag_list:
        frag = QubitOperator()
        frag.terms = pkl_frag
        frag_list.append(frag)
    return frag_list, np.array(pkl_shot_alloc_arr), final_variance, np.array(pkl_cvec)


def ics_interface(data,  # obtained from prepare_main
                  ref2, ref2_name,
                  num_workers, anticommute,
                  shift_opt_lvl):
    mol_name, transform, ref1 = data['mol_name'], data['transform'], data['ref']

    # Check the second reference state
    if ref2_name == "CISD":
        mol, p_const, f_const, time_step, n_qksd = (data['mol'], data['p_const'], data['f_const'],
                                                    data['delta_t'], data['n_qksd'])
        cisd_energy = mol.cisd_energy - p_const - f_const
        phase_list = np.array([time_step * cisd_energy * k for k in range(n_qksd)])
    elif ref2_name[:4] == "TRUE":
        phase_list = None
    else:
        raise ValueError

    # Check the shifting
    if shift_opt_lvl is not None:
        _, pham, _ = data["shift_dict"][shift_opt_lvl]
    else:
        pham = data['pham']

    # Check if previous results exist
    prev_results = list()
    if ref2_name == "CISD":
        prev_results: List[Optional[Any]] = [None for _ in range(n_qksd)]
        for k in range(n_qksd):
            path_name = path_ics(mol_name, transform, anticommute, shift_opt_lvl, ref2_name + f"_{k}")
            if os.path.isfile(path_name):
                with open(path_name, "rb") as f:
                    prev_results[k] = unpickle_ics(*pickle.load(f))
            elif ref2 is None:
                raise ValueError
        if all([r is not None for r in prev_results]):
            return prev_results

    elif ref2_name[:4] == "TRUE":
        path_name = path_ics(mol_name, transform, anticommute, shift_opt_lvl, ref2_name)
        if os.path.isfile(path_name):
            with open(path_name, "rb") as f:
                return unpickle_ics(*pickle.load(f))
        elif ref2 is None:
            raise ValueError
    else:
        raise AssertionError

    print(f"Cov Pauli starts at {datetime.now()}")
    t = time()
    cov_buf_dir = path_cov(mol_name, transform, anticommute, shift_opt_lvl, ref2_name)
    initial_grp, cov_dict = init_ics(pham, ref1, ref2, num_workers, anticommute,
                                     cov_buf_dir=cov_buf_dir,
                                     phase_list=phase_list)
    t = time() - t
    print(f"Cov Pauli Took {t} secs")
    print(f"ICS starts at {datetime.now()}")
    if ref2_name == "CISD":
        for k, ph in tqdm(enumerate(phase_list)):
            if prev_results[k] is None:
                grp_ham, frag_shots, final_variance, cvec = run_ics(pham, initial_grp, cov_dict[ph],
                                                                    transition=True, conv_atol=1e-5, conv_rtol=1e-3,
                                                                    debug=False)
                prev_results[k] = (grp_ham, frag_shots, final_variance, cvec)
                path_name = path_ics(mol_name, transform, anticommute, shift_opt_lvl, ref2_name + f"_{k}")
                with open(path_name, "wb") as f:
                    pickle.dump(pickle_ics(*prev_results[k]), f)
        return prev_results
    elif ref2_name[:4] == "TRUE":
        grp_ham, frag_shots, final_variance, cvec = run_ics(pham, initial_grp, cov_dict,
                                                            transition=True, conv_atol=1e-5, conv_rtol=1e-3,
                                                            debug=False)
        path_name = path_ics(mol_name, transform, anticommute, shift_opt_lvl, ref2_name)
        with open(path_name, "wb") as f:
            pickle.dump(pickle_ics(grp_ham, frag_shots, final_variance, cvec), f)
        return grp_ham, frag_shots, final_variance, cvec
    else:
        raise AssertionError
