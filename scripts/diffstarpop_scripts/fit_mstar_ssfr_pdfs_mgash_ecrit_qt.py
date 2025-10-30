import os
import h5py
import numpy as np
from jax import (
    numpy as jnp,
    jit as jjit,
    random as jran,
    grad,
    vmap,
)
import argparse
from time import time
from scipy.optimize import minimize
from jax.example_libraries import optimizers as jax_opt

from collections import OrderedDict, namedtuple

from diffstar.defaults import TODAY, LGT0
from diffmah.diffmah_kernels import mah_halopop

from diffstar.diffstarpop.loss_kernels.mstar_ssfr_loss_mgash_ecrit_qt import (
    loss_mstar_kern_tobs_grad_wrapper,
    loss_mstar_ssfr_kern_tobs_grad_wrapper,
    loss_combined_wrapper,
    loss_combined_3loss_wrapper,
)

from diffstar.diffstarpop.kernels.defaults_mgash_ecrit_qt import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
    get_bounded_diffstarpop_params,
)

from fit_get_loss_helpers_mgash_anyz import (
    get_loss_data_pdfs_mstar,
    get_loss_data_pdfs_ssfr_central,
    get_loss_data_pdfs_ssfr_satellite,
)
from diffstar.diffstarpop.kernels.params import (
    DiffstarPop_UParams_Diffstarfits_mgash_ecrit_qt,
)

BEBOP_SMHM_MEAN_DATA = "/lcrc/project/halotools/alarcon/results/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-indir", help="input drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "-outdir", help="output drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "-nhalos", help="Number of halos for fitting", type=int, default=100
    )
    parser.add_argument(
        "-nstep", help="Number of steps for fitting", type=int, default=1000
    )
    parser.add_argument(
        "-outname",
        help="output fname for best params",
        type=str,
        default="bestfit_diffstarpop_params",
    )
    parser.add_argument(
        "-loss_type",
        help="Which data to target",
        type=str,
        choices=["mstar", "mstar_ssfr_cen", "mstar_ssfr_cen_sat"],
        default="mstar",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="Path were diffstarpop params are stored",
    )
    parser.add_argument(
        "--print_loss",
        type=int,
        default=100,
        help="How many steps before printing current loss",
    )

    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    nhalos = args.nhalos
    n_step = args.nstep
    outname = args.outname
    params_path = args.params_path
    loss_type = args.loss_type

    # Load MStar pdf data ---------------------------------------------

    if loss_type == "mstar":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data = (loss_data_mstar,)
    elif loss_type == "mstar_ssfr_cen":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data_ssfr_cen, plot_data_ssfr_cen = get_loss_data_pdfs_ssfr_central(
            indir, nhalos
        )
        loss_data = (loss_data_mstar, loss_data_ssfr_cen)
    elif loss_type == "mstar_ssfr_cen_sat":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data_ssfr_cen, plot_data_ssfr_cen = get_loss_data_pdfs_ssfr_central(
            indir, nhalos
        )
        loss_data_ssfr_sat, plot_data_ssfr_sat = get_loss_data_pdfs_ssfr_satellite(
            indir, nhalos
        )
        loss_data = (loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat)

    # Define loss kernel ---------------------------------------------
    if loss_type == "mstar":
        loss_kernel = loss_mstar_kern_tobs_grad_wrapper
    elif loss_type == "mstar_ssfr_cen":
        loss_kernel = loss_combined_wrapper
    elif loss_type == "mstar_ssfr_cen_sat":
        loss_kernel = loss_combined_3loss_wrapper

    # Register params ---------------------------------------------

    if params_path is None:
        all_u_params = jnp.asarray(DEFAULT_DIFFSTARPOP_U_PARAMS)
    elif params_path.startswith("diffstarfits"):
        sim_name = params_path.split("_")[1:]
        sim_name = ("_").join(sim_name)
        params_tuple = DiffstarPop_UParams_Diffstarfits_mgash_ecrit_qt[sim_name]
        all_u_params = jnp.asarray(params_tuple)
    else:
        params = np.load(params_path)
        all_u_params = params["diffstarpop_u_params"]

    # Run fitter ---------------------------------------------
    print("Running fitter...")

    params_init = jnp.asarray(all_u_params)
    loss_kernel(params_init, *loss_data)

    start = time()

    step_size = 0.01

    loss_arr = np.zeros(n_step).astype("f4") + np.inf

    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    n_mah = 100

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    no_nan_grads_arr = np.zeros(n_step)
    for istep in range(n_step):
        start = time()
        ran_key, subkey = jran.split(ran_key, 2)

        p = np.array(get_params(opt_state))

        loss, grads = loss_kernel(p, *loss_data)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        if ~no_nan_loss:
            print("NaN in loss, trying to take extra gradient step")
            opt_state2 = opt_update(istep, current_grads, opt_state)
            p2 = np.array(get_params(opt_state2))
            loss2, grads2 = loss_kernel(p2, *loss_data)
            no_nan_params2 = np.all(np.isfinite(p2))
            no_nan_loss2 = np.isfinite(loss2)
            no_nan_grads2 = np.all(np.isfinite(grads2))
            if ~no_nan_params2 | ~no_nan_loss2 | ~no_nan_grads2:
                print("Extra step failed")
                continue
            else:
                p = p2.copy()
                loss = loss2.copy()
                grads = grads2.copy()
                no_nan_params = np.all(np.isfinite(p))
                no_nan_loss = np.isfinite(loss)
                no_nan_grads = np.all(np.isfinite(grads))

        if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
            # break
            if istep > 0:
                indx_best = np.nanargmin(loss_arr[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
        else:
            params_arr[istep, :] = p
            loss_arr[istep] = loss
            opt_state = opt_update(istep, grads, opt_state)

        current_grads = grads.copy()

        no_nan_grads_arr[istep] = ~no_nan_grads
        end = time()
        if istep % args.print_loss == 0:
            print(istep, loss, end - start, no_nan_grads)
        if ~no_nan_grads:
            break

    argmin_best = np.argmin(loss_arr)
    best_fit_u_params = params_arr[argmin_best]

    def return_params_from_result(best_fit_u_params):
        bestfit_u_tuple = DEFAULT_DIFFSTARPOP_U_PARAMS._make(best_fit_u_params)
        diffstarpop_params = get_bounded_diffstarpop_params(bestfit_u_tuple)
        return diffstarpop_params

    best_fit_params = return_params_from_result(best_fit_u_params)
    best_fit_params = jnp.asarray(best_fit_params)

    np.savez(
        os.path.join(outdir, outname) + ".npz",
        diffstarpop_params=best_fit_params,
        diffstarpop_u_params=best_fit_u_params,
    )
