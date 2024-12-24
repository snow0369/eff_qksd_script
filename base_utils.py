import os
import pickle
from time import time
from typing import Tuple, List

import numpy as np
from openfermion import FermionOperator, get_fermion_operator, QubitOperator, MolecularData

from ofex.hamiltonian import PolyacenePPP
from ofex.linalg.sparse_tools import diagonalization, state_dot, expectation
from ofex.measurement import sorted_insertion
from ofex.measurement.killer_shift import killer_shift_opt_fermion_hf
from ofex.propagator import exact_rte, trotter_rte_by_si_ref
from ofex.state.chem_ref_state import hf_ground, cisd_ground
from ofex.state.state_tools import get_num_qubits
from ofex.transforms import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.utils.chem import molecule_example, run_driver
from ofex_algorithms.qksd.qksd_simulation import ideal_qksd_toeplitz, ideal_qksd_nontoeplitz
from ofex_algorithms.qksd.qksd_utils import toeplitz_arr_to_mat


DEFAULT_DRIVER = "pyscf"

DIR_HAMILTONIAN = './buffer/chemistry_data/hamiltonian/'
DIR_REFSTATE = "./buffer/chemistry_data/reference_state/"
DIR_SPECTRUM = "./buffer/chemistry_data/spectrum"
DIR_PROPAGATOR = "./buffer/chemistry_data/propagator"

DIR_QKSD = "./buffer/chemistry_qksd_data/"
DIR_PROBBUF = "./buffer/qksd_prob/"


def rec_path(dir_list: Tuple[str, ...], fname, tag):
    d: str = os.path.join(*dir_list)
    if not os.path.isdir(d):
        os.makedirs(d)
    if tag is None:
        return os.path.join(d, fname)
    else:
        fname = '.'.join(fname.split('.')[:-1]) + f"_{tag}." + fname.split('.')[-1]
        return os.path.join(d, fname)


def path_fermion_hamiltonian(mol_name, tag=None):
    return rec_path((DIR_HAMILTONIAN, mol_name), "fermion.pkl", tag)


def path_pauli_hamiltonian(mol_name, transform, tag=None):
    return rec_path((DIR_HAMILTONIAN, mol_name), f"{transform}.pkl", tag)


def path_shifted_hamiltonian(mol_name, transform, tag=None):
    return rec_path((DIR_HAMILTONIAN, mol_name), f"{transform}_shifted.pkl", tag)


def path_cisd_state(mol_name, transform, tag=None):
    return rec_path((DIR_REFSTATE, mol_name), f"cisd_{transform}.pkl", tag)


def path_spectrum_analysis(mol_name, ref_name, transform, tag=None):
    return rec_path((DIR_SPECTRUM, mol_name), f"{ref_name}_{transform}.pkl", tag)


def path_propagator(mol_name, transform, time_step_str, n_trotter, tag=None):
    if n_trotter == 0 or n_trotter is None:
        return rec_path((DIR_PROPAGATOR, mol_name, transform), f"Δt={time_step_str}_exactrte.npy", tag)
    else:
        return rec_path((DIR_PROPAGATOR, mol_name, transform), f"Δt={time_step_str}_trotter={n_trotter}.npy", tag)


def path_ideal_matrix(mol_name, transform, time_step_str, n_trotter, shift_opt_lvl, ref_name, toeplitz, tag=None):
    if n_trotter == 0 or n_trotter is None:
        return rec_path((DIR_QKSD, mol_name, transform, "ideal_mat", f"Δt={time_step_str}_exactrte"),
                     f"shift={shift_opt_lvl}_ref={ref_name}_toeplitz={toeplitz}.pkl", tag)
    else:
        return rec_path((DIR_QKSD, mol_name, transform, "ideal_mat", f"Δt={time_step_str}_trotter={n_trotter}"),
                     f"shift={shift_opt_lvl}_ref={ref_name}_toeplitz={toeplitz}.pkl", tag)

def path_probbuf(mol_name, transform, anticommute, meas_opt_name):
    meas_type = "FH" if not anticommute else "LCU"
    return rec_path((DIR_PROBBUF, f"{mol_name}_{transform}", meas_opt_name), "prob_buf.pkl", meas_type)


def prepare_hamiltonian_refstates(mol_name, transform,
                                  tag=None, load_shift=True,
                                  print_progress=False):
    if print_progress:
        print(" === Start Preparing Hamiltonian ===")

    t = time()
    if mol_name not in ["Benzene_PPP"]:
        mol = molecule_example(mol_name)
        mol.load()
        mol = run_driver(mol, run_cisd=True, run_fci=True, driver=DEFAULT_DRIVER)
    elif mol_name == "Benzene_PPP":
        mol = PolyacenePPP(n_ring=1)
    else:
        raise ValueError
    if print_progress:
        print(f" === MolecularData Done ({time() - t} sec) ===")

    t = time()
    # keyword arguments required in ofex.transforms.fermion_to_qubit_operator
    if transform == "bravyi_kitaev":
        f2q_kwargs = {"n_qubits": mol.n_qubits}
    elif transform == "symmetry_conserving_bravyi_kitaev":
        f2q_kwargs = {"active_fermions": mol.n_electrons,
                      "active_orbitals": mol.n_qubits}
    elif transform == "jordan_wigner":
        f2q_kwargs = dict()
    else:
        raise NotImplementedError

    # HF Reference State
    if isinstance(mol, MolecularData):
        ref = hf_ground(mol, fermion_to_qubit_map=transform, **f2q_kwargs)
        f_ref = hf_ground(mol, fermion_to_qubit_map=None)
    elif isinstance(mol, PolyacenePPP):
        f_ref = mol.hf_state()
        ref = fermion_to_qubit_state(f_ref, transform, **f2q_kwargs)
    else:
        raise TypeError

    # PySCF results may be different for every run. Thus, the objects need to be pickled.
    # Prepare Fermion Hamiltonian
    fham_path = path_fermion_hamiltonian(mol_name, tag)
    if os.path.isfile(fham_path):
        with open(fham_path, 'rb') as f:
            fham_t, f_const = pickle.load(f)
        fham = FermionOperator()
        fham.terms = fham_t
        if print_progress:
            print(f" === Fermion Hamiltonian Loaded from {fham_path} ===")
    else:
        fham = mol.get_molecular_hamiltonian()
        if not isinstance(fham, FermionOperator):
            fham = get_fermion_operator(fham)
        f_const = fham.constant
        fham = fham - f_const
        with open(fham_path, 'wb') as f:
            pickle.dump((fham.terms, f_const), f)
        if print_progress:
            print(f" === Fermion Hamiltonian Evaluated and Saved to {fham_path} ===")

    # Prepare Pauli Hamiltonian
    pham_path = path_pauli_hamiltonian(mol_name, transform, tag)
    if os.path.isfile(pham_path):
        with open(pham_path, 'rb') as f:
            pham_t, p_const = pickle.load(f)
        pham = QubitOperator()
        pham.terms = pham_t
        if print_progress:
            print(f" === Qubit Hamiltonian Loaded from {pham_path} ===")
    else:
        pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
        p_const = pham.constant
        pham = pham - p_const
        with open(pham_path, 'wb') as f:
            pickle.dump((pham.terms, p_const), f)
        if print_progress:
            print(f" === Qubit Hamiltonian Evaluated and Saved to {pham_path} ===")

    if print_progress:
        print(f" === Hamiltonian Preparation Done ({time() - t}) ===")

    # Prepare CISD State
    cisd_state_path = path_cisd_state(mol_name, transform, tag)
    if os.path.isfile(cisd_state_path):
        with open(cisd_state_path, 'rb') as f:
            cisd_state = pickle.load(f)
        if print_progress:
            print(f" === CISD state Loaded from {cisd_state_path} ===")
    elif isinstance(mol, MolecularData):
        cisd_state = cisd_ground(mol)
        cisd_state = fermion_to_qubit_state(cisd_state, transform, **f2q_kwargs)
        with open(cisd_state_path, 'wb') as f:
            pickle.dump(cisd_state, f)
        if print_progress:
            print(f" === CISD state Evaluated and Saved to {cisd_state_path} ===")
    elif isinstance(mol, PolyacenePPP):
        cisd_state = None
    else:
        raise TypeError

    # Perform shifting
    if load_shift:
        shift_path = path_shifted_hamiltonian(mol_name, transform, tag)
        if os.path.isfile(shift_path):
            with open(shift_path, 'rb') as f:
                shift_dict = pickle.load(f)
            for opt_lv, (fh_t, ph_t, c) in shift_dict.items():
                fh, ph = FermionOperator(), QubitOperator()
                fh.terms, ph.terms = fh_t, ph_t
                shift_dict[opt_lv] = (fh, ph, c)
            if print_progress:
                print(f" === Shifted Hamiltonian Loaded from {shift_path} ===")
        else:
            if print_progress:
                print(f" === Start Shifting ===")
            shift_dict = dict()
            repeat_opt = 1
            hf_vector = list(f_ref.keys())[0]
            for optimization_level in [0, 1, 2]:
                t = time()
                if print_progress:
                    print(f"opt_level={optimization_level} ", end='')
                shift_fham, shift_pham, shift_const = killer_shift_opt_fermion_hf(fham, hf_vector, transform,
                                                                                  optimization_level,
                                                                                  repeat_opt,
                                                                                  f2q_kwargs)
                shift_dict[optimization_level] = (shift_fham, shift_pham, shift_const)
                if print_progress:
                    print(f"| norm = {shift_pham.induced_norm(order=2)} | ({time() - t} sec)")
            shift_op_pikl = {oplvl: (fh.terms, ph.terms, c) for oplvl, (fh, ph, c) in shift_dict.items()}
            with open(shift_path, 'wb') as f:
                pickle.dump(shift_op_pikl, f)
    else:
        shift_dict = None

    n_qubits = get_num_qubits(ref)

    return {"pham": pham, "fham": fham, "p_const": p_const, "f_const": f_const,
            "ref": ref, "f_ref": f_ref, "cisd_state": cisd_state,
            "shift_dict": shift_dict,
            "mol_name": mol_name, "transform": transform,
            "mol": mol, "n_qubits": n_qubits, "f2q_kwargs": f2q_kwargs}


def spectrum_analysis(pham, ref, mol_name, transform,
                      ref_name="HF", tag=None, atol=1e-8,
                      print_progress=False, **_)\
        -> List[Tuple[int, complex, complex]]:
    spectrum_path = path_spectrum_analysis(mol_name, ref_name, transform, tag)
    if os.path.isfile(spectrum_path):
        with open(spectrum_path, 'rb') as f:
            eigval_overlap_pair = pickle.load(f)
        if print_progress:
            print(f" === Spectrum Analysis Loaded from {spectrum_path} ===")
    else:
        if print_progress:
            print(" === Spectrum Analysis Started ... ", end="")
        t = time()
        n_qubits = get_num_qubits(ref)
        w, v = diagonalization(pham, n_qubits, sparse_eig=False)
        idx_order = np.argsort(w)
        eigval_overlap_pair = list()
        for idx_eig in idx_order:
            eigval, eigvec = w[idx_eig], v[:, idx_eig]
            overlap = state_dot(ref, eigvec)
            if abs(overlap) ** 2 > atol:
                eigval_overlap_pair.append((idx_eig, eigval, overlap))
        with open(spectrum_path, 'wb') as f:
            pickle.dump(eigval_overlap_pair, f)
        if print_progress:
            print(f" Done ({time() - t} sec) ===")

    return eigval_overlap_pair


def real_time_propagator(mol_name, transform, pham, time_step, n_qubits, n_trotter,
                         tag=None, n_digits_t=6,
                         save=True, print_progress=False, **_):
    if time_step <= 0:
        raise ValueError
    time_step = round(time_step, n_digits_t - int(np.floor(np.log10(abs(time_step)))) - 1)
    time_step_str = ("{" + f":.{n_digits_t - 1}e" + "}").format(time_step)

    prop_path = path_propagator(mol_name, transform, time_step_str, n_trotter, tag)
    if os.path.isfile(prop_path):
        prop = np.load(prop_path)
        if print_progress:
            print(f" === Propagator Loaded from {prop_path} ===")
    else:
        if print_progress:
            print(f" === Propagator Evaluation Started ... ", end="")
        t = time()
        if n_trotter == 0 or n_trotter is None:
            prop = exact_rte(pham, time_step, exact_sparse=True)
        else:
            prop = trotter_rte_by_si_ref(pham, time_step, n_qubits, n_trotter, exact_sparse=False)
        if save:
            np.save(prop_path, prop)
            if print_progress:
                print(f" Done({time() - t} sec) and saved to {prop_path} ===")
        elif print_progress:
            print(f" Done({time() - t} sec) ===")
    return prop


def ideal_qksd_matrix(mol_name, transform, pham, time_step, n_qubits, shift_dict, ref,
                      shift_opt_lvl, n_krylov, n_trotter,
                      tag=None, ref_name="HF", toeplitz=True,
                      n_digits_t=6, save_prop=True, print_progress=False, **_):
    # If you run this multiple times with different n_krylov, make sure larger n_krylov is given first.
    if time_step <= 0:
        raise ValueError
    time_step = round(time_step, n_digits_t - int(np.floor(np.log10(abs(time_step)))) - 1)
    time_step_str = ("{" + f":.{n_digits_t - 1}e" + "}").format(time_step)

    ideal_qksd_matrix_path = path_ideal_matrix(mol_name, transform, time_step_str, n_trotter, shift_opt_lvl,
                                               ref_name, toeplitz, tag)
    if os.path.isfile(ideal_qksd_matrix_path):
        with open(ideal_qksd_matrix_path, 'rb') as f:
            h_mat, s_mat, shift_const = pickle.load(f)
        h_mat, s_mat = np.array(h_mat), np.array(s_mat)
        read_n_krylov = h_mat.shape[0]
        if read_n_krylov >= n_krylov:
            if print_progress:
                print(f" === Ideal Matrix Loaded from {ideal_qksd_matrix_path} ===")
            return h_mat[:n_krylov, :n_krylov], s_mat[:n_krylov, :n_krylov], shift_const

    # Requires larger n_krylov than saved data.
    if print_progress:
        print(" === Propagator Called from Ideal Matrix ===")
    prop = real_time_propagator(mol_name, transform, pham, time_step, n_qubits, n_trotter, tag, n_digits_t,
                                save_prop, print_progress)

    if shift_opt_lvl is not None:
        _, pham, shift_const = shift_dict[shift_opt_lvl]
    else:
        shift_const = 0.0

    if print_progress:
        print(" === Ideal QKSD Matrix Evaluation Started ... ", end="")
    t = time()
    if toeplitz:
        h_mat, s_mat = ideal_qksd_toeplitz(pham, prop, ref, n_krylov)
    else:
        h_mat, s_mat = ideal_qksd_nontoeplitz(pham, prop, ref, n_krylov)

    with open(ideal_qksd_matrix_path, 'wb') as f:
        pickle.dump((h_mat.tolist(), s_mat.tolist(), shift_const), f)

    if print_progress:
        print(f" Done({time() - t} sec) and saved to {ideal_qksd_matrix_path} ===")

    return h_mat, s_mat, shift_const


def approx_sampling_noise(tot_shots, n_batch, h_norm, s_mat, toeplitz=True, is_lcu=False):
    n = s_mat.shape[0]
    if is_lcu:
        raise NotImplementedError
    if toeplitz:
        shots_per_element = tot_shots / (2 * (n - 1))
        std_h = h_norm / np.sqrt(shots_per_element)
        noise_h = np.zeros((n_batch, n), dtype=complex)
        noise_h[:, 1:] = np.random.normal(0, std_h, (n_batch, n - 1)) \
                         + 1j * np.random.normal(0, std_h, (n_batch, n - 1))
        noise_h = toeplitz_arr_to_mat(noise_h)
    else:
        shots_per_element = tot_shots / (n * (n - 1))
        std_h = h_norm / np.sqrt(shots_per_element)
        noise_h = np.zeros((n_batch, n, n), dtype=complex)
        for i in range(n - 1):
            noise_h[:, i, (i + 1):] = np.random.normal(0, std_h, (n_batch, n - i - 1)) \
                                      + 1j * np.random.normal(0, std_h, (n_batch, n - i - 1))
            noise_h[:, (i + 1):, i] = noise_h[:, i, (i + 1):].conj()

    shots_per_element = tot_shots / (2 * (n - 1))
    noise_s = np.zeros((n_batch, n), dtype=complex)
    for i in range(1, n):
        re, im = s_mat[0, i].real, s_mat[0, i].imag
        if not (abs(re) < 1 and abs(im) < 1):
            raise ValueError
        std_s_re = np.sqrt((1 - re ** 2) / (4 * shots_per_element))
        std_s_im = np.sqrt((1 - im ** 2) / (4 * shots_per_element))
        noise_s[:, i] = np.random.normal(0, std_s_re, n_batch)
        noise_s[:, i] += 1j * np.random.normal(0, std_s_im, n_batch)

    noise_s = toeplitz_arr_to_mat(noise_s)

    return noise_h, noise_s


def approx_sampling_noise_norm(tot_shots, n, h_norm, toeplitz=True, is_lcu=False):
    if is_lcu:
        raise NotImplementedError
    norm_s_toeplitz = 2 * n * np.sqrt(2 * np.log(2 * n) / tot_shots)
    if toeplitz:
        return h_norm * norm_s_toeplitz, norm_s_toeplitz
    else:
        norm_h_non_toeplitz = 2 * h_norm * n * np.sqrt(n * np.log(2 * n) / tot_shots)
        return norm_h_non_toeplitz, norm_s_toeplitz

def approx_sampling_noise_norm_s(tot_shots, n, is_lcu=False):
    if is_lcu:
        tot_shots /= 2
    return 2 * n * np.sqrt(2 * np.log(2 * n) / tot_shots)


def prepare_main(mol_name,
                 transform,
                 n_trotter,
                 n_qksd,
                 shift_opt_lvl,
                 gap_estimation:str ="energy"):
    if gap_estimation not in ["overlap", "energy"]:
        raise ValueError

    data = prepare_hamiltonian_refstates(mol_name, transform, print_progress=True)

    spectrum = spectrum_analysis(**data, print_progress=True)  # List of eigidx, eigval, overlap
    sorted_spectrum_energy = sorted(spectrum, key=lambda x: x[1])  # Sort by eigval
    idx_eig, gnd_energy, gnd_overlap = sorted_spectrum_energy[0]
    assert idx_eig == 0

    sorted_spectrum_overlap = sorted(spectrum, key=lambda x: abs(x[2]), reverse=True)
    gnd_overlap = abs(gnd_overlap) ** 2
    if sorted_spectrum_overlap[0][0] == 0:  # Gnd state overlaps at most.
        spectral_gap_overlap = sorted_spectrum_overlap[1][1] - gnd_energy
        second_overlap = abs(sorted_spectrum_overlap[1][2]) ** 2
    else:  # Find the largest overlap
        for idx, second_energy, second_overlap in sorted_spectrum_energy[1:]:
            second_overlap = abs(second_overlap) ** 2
            if second_overlap > gnd_overlap / np.e:
                break
        spectral_gap_overlap = second_energy - gnd_energy

    if gap_estimation == "energy":
        spectral_gap = sorted_spectrum_energy[1][1] - gnd_energy
    elif gap_estimation == "overlap":
        spectral_gap = spectral_gap_overlap
    else:
        raise AssertionError
    assert np.isclose(spectral_gap.imag, 0.0)
    spectral_gap = spectral_gap.real

    print("")
    if isinstance(data["mol"], MolecularData):
        hf_energy = data["mol"].hf_energy - data["p_const"] - data["f_const"]
        fci_energy = data["mol"].fci_energy - data["p_const"] - data["f_const"]
    elif isinstance(data["mol"], PolyacenePPP):
        hf_energy = expectation(data['pham'], data['ref'], sparse=True)
        fci_energy = diagonalization(data['pham'], data['n_qubits'], sparse_eig=True, n_eigen=1)[0][0]
    else:
        raise TypeError
    print(f" E_corr = {hf_energy - fci_energy}")
    print(f" E0 = {gnd_energy}  |γ0|2 = {gnd_overlap}")
    print(f" E1 - E0 = {spectral_gap}  |γ1|2 = {second_overlap}")

    pham_norm = sum([frag.induced_norm(order=2) for frag in sorted_insertion(data["pham"], anticommute=True)])
    if shift_opt_lvl is not None:
        shift_fham, shift_pham, _ = data["shift_dict"][shift_opt_lvl]
        meas_norm = sum([frag.induced_norm(order=2)
                         for frag in sorted_insertion(shift_pham, anticommute=False)])
    else:
        meas_norm = sum([frag.induced_norm(order=2) for frag in sorted_insertion(data["pham"], anticommute=False)])

    delta_t = np.pi / spectral_gap
    print(f" Δt = {delta_t}  n = {n_qksd}")
    # print(f" ‖H‖(meas) = {meas_norm} ")
    print(f" ‖H‖(lcu)  = {pham_norm} ")
    print("")

    print(f" n = {n_qksd}")
    propagator = real_time_propagator(mol_name, transform, data["pham"], delta_t, data["n_qubits"], n_trotter,
                                      save=True)
    h_mat, s_mat, shift_const = ideal_qksd_matrix(n_krylov=n_qksd, shift_opt_lvl=shift_opt_lvl,
                                                  n_trotter=n_trotter,
                                                  time_step=delta_t,
                                                  print_progress=True, **data)
    ret_dict = {
        "spectrum": spectrum,
        "spectral_gap": spectral_gap,
        "hf_energy": hf_energy,
        "fci_energy": fci_energy,
        "gnd_energy": gnd_energy,
        "gnd_overlap": gnd_overlap,
        "pham_norm": pham_norm,
        "propagator": propagator,
        # "meas_norm": meas_norm,
        "delta_t": delta_t,
        "n_qksd": n_qksd,
        "h_mat": h_mat,
        "s_mat": s_mat,
        "shift_const": shift_const
    }

    assert len(set(ret_dict.keys()) & set(data.keys())) == 0

    return {**ret_dict, **data}
