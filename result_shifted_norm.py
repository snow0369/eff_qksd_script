import csv

from ofex.measurement import sorted_insertion
from base_utils import prepare_hamiltonian_refstates
from result_base_setting import default_transform, default_mol_list, default_shift_opt_lvl


def calc_meas_norm(frag_list):
    return sum([frag.induced_norm(order=2) for frag in frag_list])

def shifted_norms(mol_name, tag=None, print_progress=True):
    data = prepare_hamiltonian_refstates(mol_name, default_transform, tag,
                                         load_shift=True, print_progress=print_progress)
    # Get non-shifted norm
    norm_fh = calc_meas_norm(sorted_insertion(data["pham"], anticommute=False))
    norm_lcu = calc_meas_norm(sorted_insertion(data["pham"], anticommute=True))

    # Get shifted norm
    _, shift_pham, _ = data["shift_dict"][default_shift_opt_lvl]
    norm_shift_fh = calc_meas_norm(sorted_insertion(shift_pham, anticommute=False))
    norm_shift_lcu = calc_meas_norm(sorted_insertion(shift_pham, anticommute=True))

    red_factor_fh = (1.0 - norm_shift_fh / norm_fh) * 100
    red_factor_lcu = (1.0 - norm_shift_lcu / norm_lcu) * 100

    print("========================== RESULT ==============================")
    print(f"{mol_name}")
    print(f"FH")
    print(f"\tNon-shifted Norm : {norm_fh}")
    print(f"\tShifted Norm     : {norm_shift_fh}")
    print(f"\tReduction Factor : {red_factor_fh:.2f}%")

    print(f"LCU")
    print(f"\tNon-shifted Norm: {norm_lcu}")
    print(f"\tShifted Norm    : {norm_shift_lcu}")
    print(f"\tReduction Factor: {red_factor_lcu:.2f}%")
    print("================================================================")


    return (norm_fh, norm_shift_fh, red_factor_fh,
            norm_lcu, norm_shift_lcu, red_factor_lcu)

def generate_csv(mol_list):
    with open("result_shifted_norm.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Molecule", "FH Norm", "Shifted FH Norm", "FH Reduction Factor(%)",
                         "LCU Norm", "Shifted LCU Norm", "LCU Reduction Factor(%)"])
        for mol_name in mol_list:
            norm_fh, norm_shift_fh, red_factor_fh, \
            norm_lcu, norm_shift_lcu, red_factor_lcu = shifted_norms(mol_name)
            writer.writerow([mol_name, norm_fh, norm_shift_fh, red_factor_fh,
                             norm_lcu, norm_shift_lcu, red_factor_lcu,])

if __name__ == "__main__":
    generate_csv(default_mol_list)
