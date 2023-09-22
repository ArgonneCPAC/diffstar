"""This script generates the unit-testing data used to freeze the behavior of Diffstar.

The purpose of these testing data is to ensure that if the behavior of Diffstar changes
as the code evolves, then we will find out about it.

"""
import argparse
import os

import numpy as np
from diffmah.individual_halo_assembly import DEFAULT_MAH_PARAMS, _calc_halo_history

from diffstar.data_loaders.load_bpl import LGT0 as LGT0_BPL
from diffstar.data_loaders.load_bpl import TASSO_BPL_DRN, load_bpl_diffstar_data
from diffstar.fitting_helpers.fitting_kernels import _sfr_history_from_mah
from diffstar.fitting_helpers.tests.test_fitting_kernels_are_frozen import (
    _get_default_mah_params,
    calc_sfh_on_default_params,
)
from diffstar.utils import _jax_get_dt_array

MAH_K = DEFAULT_MAH_PARAMS["mah_k"]


TASSO = "/Users/aphearin/work/repositories/python/diffstar/diffstar/tests/testing_data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testing_drn", help="Path to testing_data directory")
    args = parser.parse_args()
    testing_drn = args.testing_drn

    # Calculate SFH for default parameters
    sfh_args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = sfh_args

    default_mah_params = _get_default_mah_params()

    np.savetxt(os.path.join(testing_drn, "default_params_test_lgt.txt"), lgt)
    np.savetxt(os.path.join(testing_drn, "default_params_test_dt.txt"), dt)
    np.savetxt(os.path.join(testing_drn, "default_params_test_dmhdt.txt"), dmhdt)
    np.savetxt(os.path.join(testing_drn, "default_params_test_log_mah.txt"), log_mah)
    np.savetxt(
        os.path.join(testing_drn, "default_params_test_u_ms_params.txt"), u_ms_params
    )
    np.savetxt(
        os.path.join(testing_drn, "default_params_test_u_q_params.txt"), u_q_params
    )
    np.savetxt(os.path.join(testing_drn, "default_params_test_sfh.txt"), sfh)
    np.savetxt(
        os.path.join(testing_drn, "default_params_test_diffmah_params.txt"),
        default_mah_params,
    )

    # Calculate SFH for 4 example BPL subhalos
    bpl, t_bpl, all_param_colnames = load_bpl_diffstar_data(TASSO_BPL_DRN)
    lgt_bpl = np.log10(t_bpl)
    dt_bpl = _jax_get_dt_array(t_bpl)
    mah_colnames, u_ms_colnames, u_q_colnames = all_param_colnames

    halo_ids_test_sample = np.array([2811277077, 2810821061, 2810385737, 2819082853])
    indx_test = np.array(
        [np.argmin(np.abs(bpl["halo_id"] - hid)) for hid in halo_ids_test_sample]
    )
    sample = bpl[indx_test]

    mah_params_test_sample = np.array([sample[key] for key in mah_colnames]).T
    ms_u_params_test_sample = np.array([sample[key] for key in u_ms_colnames]).T
    q_u_params_test_sample = np.array([sample[key] for key in u_q_colnames]).T

    sfh_test_sample = []
    for ih in range(mah_params_test_sample.shape[0]):
        all_mah_params_ih = np.array(
            (
                LGT0_BPL,
                mah_params_test_sample[ih, 0],
                mah_params_test_sample[ih, 1],
                MAH_K,
                mah_params_test_sample[ih, 2],
                mah_params_test_sample[ih, 3],
            )
        )
        ms_u_params_ih = np.array(ms_u_params_test_sample[ih, :])
        q_u_params_ih = np.array(q_u_params_test_sample[ih, :])

        dmhdt_ih, log_mah_ih = _calc_halo_history(lgt_bpl, *all_mah_params_ih)
        sfh_ih = _sfr_history_from_mah(
            lgt_bpl, dt_bpl, dmhdt_ih, log_mah_ih, ms_u_params_ih, q_u_params_ih
        )

        sfh_test_sample.append(sfh_ih)
    sfh_test_sample = np.array(sfh_test_sample)

    np.savetxt(os.path.join(testing_drn, "sfh_test_sample.txt"), sfh_test_sample)
    np.savetxt(
        os.path.join(testing_drn, "mah_params_test_sample.txt"), mah_params_test_sample
    )
    np.savetxt(
        os.path.join(testing_drn, "ms_u_params_test_sample.txt"),
        ms_u_params_test_sample,
    )
    np.savetxt(
        os.path.join(testing_drn, "q_u_params_test_sample.txt"), q_u_params_test_sample
    )
    np.savetxt(
        os.path.join(testing_drn, "halo_ids_test_sample.txt"), halo_ids_test_sample
    )
    np.savetxt(os.path.join(testing_drn, "lgt_bpl.txt"), lgt_bpl)
    np.savetxt(os.path.join(testing_drn, "dt_bpl.txt"), dt_bpl)
    np.savetxt(
        os.path.join(testing_drn, "halo_ids_test_sample.txt"), halo_ids_test_sample
    )
    np.savetxt(os.path.join(testing_drn, "lgt_bpl.txt"), lgt_bpl)
    np.savetxt(os.path.join(testing_drn, "dt_bpl.txt"), dt_bpl)
