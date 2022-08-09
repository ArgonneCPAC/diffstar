"""This script generates the unit-testing data used to freeze the behavior of Diffstar.

The purpose of these testing data is to ensure that if the behavior of Diffstar changes
as the code evolves, then we will find out about it.

"""
import argparse
import os
import numpy as np
from diffstar.tests.test_diffstar_is_frozen import calc_sfh_on_default_params
from diffstar.tests.test_diffstar_is_frozen import _get_default_mah_params


TASSO = "/Users/aphearin/work/repositories/python/diffstar/diffstar/tests/testing_data"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("testing_drn", help="Path to testing_data directory")
    args = parser.parse_args()
    testing_drn = args.testing_drn

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
