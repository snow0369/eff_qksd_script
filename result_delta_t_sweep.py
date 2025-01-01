import os

import numpy as np
import matplotlib.pyplot as plt

from base_utils import prepare_main
from result_base_setting import default_transform, default_n_trotter, default_shift_opt_lvl, default_n_qksd

from ofex.propagator import exact_rte
from ofex_algorithms.qksd.qksd_simulation import ideal_qksd_toeplitz
from ofex_algorithms.qksd.qksd_utils import trunc_eigh_verbose

def sweep_time_step_cond_number(data, overwrite=False):
    fname = "./result_delta_t_sweep.csv"
    if os.path.exists(fname) and not overwrite:
        return

    gnd_energy = data["gnd_energy"]
    pham, pham_norm = data["pham"], data["pham_norm"]
    ref, n_qksd = data["ref"], data["n_qksd"]

    pham_prop = pham / pham_norm

    min_delta_t, max_delta_t = 0.1, 10.0
    n_step = 100

    prop_rte_0 = exact_rte(pham_prop, min_delta_t * np.pi)
    prop_rte = None

    cond_s_result = dict()
    convergence_result = dict()
    print(" === Start Estimation === ")
    for delta_t in np.linspace(min_delta_t, max_delta_t, n_step):
        if delta_t == min_delta_t:
            prop_rte = prop_rte_0
        else:
            prop_rte = prop_rte @ prop_rte_0
        print(f"delta_t = {delta_t}")
        ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_qksd)
        val, _, _, _, _, trunc_ideal_s = trunc_eigh_verbose(ideal_h, ideal_s, epsilon=1e-14)
        ideal_gnd = np.min(val)

        cond_s = np.linalg.cond(trunc_ideal_s)
        print(f"\t\tcond(S) = {cond_s:.5E}")
        print(f"\t\tconverg = {(ideal_gnd - gnd_energy):.5E}")
        cond_s_result[delta_t] = cond_s
        convergence_result[delta_t] = (ideal_gnd - gnd_energy)

    with open(fname, "w") as f:
        f.write("delta_t,cond(S),convergence\n")
        for delta_t in sorted(cond_s_result.keys()):
            f.write(f"{delta_t},{cond_s_result[delta_t]:.5E},{convergence_result[delta_t]:.5E}\n")

def plot_delta_t_sweep(data):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    n_qksd = data["n_qksd"]
    fname = "./result_delta_t_sweep.csv"
    delta_t_list, cond_s_result, convergence_result = list(), list(), list()
    with open(fname, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            delta_t, cond_s, convergence = line.split(",")
            delta_t_list.append(float(delta_t))
            cond_s_result.append(float(cond_s))
            convergence_result.append(float(convergence))

    fig, axes = plt.subplots(2, 1, sharex=True)
    ax_cond, ax_energy, ax_cost = axes[0], axes[0].twinx(), axes[1]

    ax_cond.plot(delta_t_list, cond_s_result, color='tab:red')
    # ax_cond.set_xlabel(r"$\Delta_t \|\hat{H}\| / \pi$")
    ax_cond.set_ylabel(r"$\mathrm{cond}(\bm{S})$", color='tab:red')
    ax_cond.set_yscale("log")
    ax_cond.tick_params(axis='y', labelcolor='tab:red')

    ax_energy.plot(delta_t_list, convergence_result, color='tab:blue')
    ax_energy.set_ylabel("$|E_0^{(n)}-E_0|$", color='tab:blue')
    ax_energy.set_yscale("log")
    ax_energy.tick_params(axis='y', labelcolor='tab:blue')

    min_e, max_e = np.min(convergence_result), np.max(convergence_result)
    ax_energy.vlines(1.0, min_e * 0.1, max_e * 10.0, linestyles="dashed", color='k')
    ax_energy.set_ylim([min_e * 0.5, max_e * 5.0])

    # Cost to make Cond(S) / sqrt(M) = 1e-3,
    # M = 1e6 * Cond(S) ** 2
    # tot_time = M * Delta_t * n * (n-1)/2
    cost = [1e6 * cond_s ** 2 * delta_t * n_qksd * (n_qksd - 1) / 2
            for delta_t, cond_s in zip(delta_t_list, cond_s_result)]
    ax_cost.plot(delta_t_list, cost, color='tab:green')
    ax_cost.set_xlabel(r"$\Delta_t \|\hat{H}\| / \pi$")
    ax_cost.set_ylabel("Total runtime", color='tab:green')
    ax_cost.tick_params(axis='y', labelcolor='tab:green')
    ax_cost.set_yscale("log")
    min_c, max_c = np.min(cost), np.max(cost)
    ax_cost.vlines(1.0, min_c * 0.1, max_c * 10.0, linestyles="dashed", color='k')
    ax_cost.set_ylim([min_c * 0.5, max_c * 5.0])

    fig.tight_layout()
    fig.savefig("./result_delta_t_sweep.png")
    plt.show()

def main():
    mol_name = "H2O"
    data = prepare_main(mol_name, default_transform, default_n_trotter, default_n_qksd[mol_name],
                        shift_opt_lvl=None, gap_estimation="overlap")
    sweep_time_step_cond_number(data, overwrite=False)
    plot_delta_t_sweep(data)

if __name__ == "__main__":
    main()
