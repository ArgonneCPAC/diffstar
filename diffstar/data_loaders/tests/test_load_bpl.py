"""
"""
import os

import pytest

from ...defaults import DEFAULT_MS_PDICT, DEFAULT_Q_PDICT
from ..load_bpl import TASSO_BPL_DRN, load_bpl_diffstar_data

try:
    assert os.path.isdir("/Users/aphearin/work/DATA/diffstar_data")
    APH_MACHINE = True
except AssertionError:
    APH_MACHINE = False


@pytest.mark.skipif(not APH_MACHINE, reason="Test only runs on APH laptop")
def test_load_bpl_returns_data_on_aph_machine():
    bpl, t_bpl, all_param_colnames = load_bpl_diffstar_data(TASSO_BPL_DRN)
    n_gals = bpl["halo_id"].size
    n_t = t_bpl.size
    assert bpl["sfrh_sim"].shape == (n_gals, n_t)

    mah_colnames, u_ms_colnames, u_q_colnames = all_param_colnames

    u_ms_colnames = [s.replace("sfh_fit_", "") for s in u_ms_colnames]
    u_q_colnames = [s.replace("sfh_fit_", "") for s in u_q_colnames]

    u_ms_colnames_v0p4 = ["u_" + s for s in DEFAULT_MS_PDICT.keys()]
    u_q_colnames_v0p4 = ["u_" + s for s in DEFAULT_Q_PDICT.keys()]
    assert set(u_ms_colnames_v0p4) == set(u_ms_colnames)
    assert set(u_q_colnames_v0p4) == set(u_q_colnames)
