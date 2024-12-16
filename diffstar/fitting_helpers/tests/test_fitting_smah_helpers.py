"""
"""

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS

from ...defaults import DEFAULT_MS_PDICT, DEFAULT_Q_PDICT
from ...utils import _jax_get_dt_array
from ..fit_smah_helpers_tpeak import get_header, get_loss_data_default

DIFFMAH_K = 3.5


def test_get_header_colnames_agree_with_model_param_names():
    header, colnames = get_header()
    assert header[0] == "#"

    assert colnames[0] == "halo_id"

    u_ms_colnames_from_header = colnames[1:6]
    ms_colnames_from_header = [s for s in u_ms_colnames_from_header]
    assert ms_colnames_from_header == list(DEFAULT_MS_PDICT.keys())

    u_q_colnames_from_header = colnames[6:10]
    q_colnames_from_header = [s for s in u_q_colnames_from_header]
    assert q_colnames_from_header == list(DEFAULT_Q_PDICT.keys())

    assert colnames[10:] == ["loss", "success"]


def test_get_loss_data_evaluates():
    t_sim = np.linspace(0.1, 13.8, 100)
    dt_sim = _jax_get_dt_array(t_sim)
    sfrh = np.random.uniform(0, 10, t_sim.size)
    smh = np.cumsum(dt_sim * sfrh) * 1e9
    log_smah_sim = np.log10(smh)

    logmp = 12.0
    p_init, loss_data = get_loss_data_default(
        t_sim, dt_sim, sfrh, log_smah_sim, logmp, DEFAULT_MAH_PARAMS
    )
