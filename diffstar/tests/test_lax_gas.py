"""
"""
import numpy as np
from .test_diffstar_is_frozen import DEFAULT_MS_PARAMS
from .test_diffstar_is_frozen import calc_sfh_on_default_params
from ..stars import SFR_PARAM_BOUNDS
from ..gas import _get_lagged_gas, _dmgas_dt_kern


def test_lax_gas_agrees_with_vmap_gas():
    tau_dep = float(DEFAULT_MS_PARAMS[-1])
    tau_dep_max = SFR_PARAM_BOUNDS["tau_dep"][3]
    _args, sfh = calc_sfh_on_default_params()
    lgtarr, dtarr, dmhdt, log_mah, u_ms_params, u_q_params = _args

    lagged_gas_args = lgtarr, dtarr, dmhdt, tau_dep, tau_dep_max
    lagged_gas = _get_lagged_gas(*lagged_gas_args)

    dt_const = dtarr.mean()

    n_t = lgtarr.size
    collector = []
    for i in range(1, n_t):
        t = 10 ** lgtarr[i - 1]
        t_table = 10 ** lgtarr[:i]
        dmhdt_table = dmhdt[:i]
        dmgdt_at_t = _dmgas_dt_kern(
            t, t_table, dt_const, dmhdt_table, tau_dep, tau_dep_max
        )
        collector.append(dmgdt_at_t)
    lax_lagged_gas = np.array(collector)

    assert np.allclose(lax_lagged_gas, lagged_gas[:-1], rtol=1e-4)
