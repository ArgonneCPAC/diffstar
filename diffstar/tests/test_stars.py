"""
"""
from jax import numpy as jnp
from diffmah.individual_halo_assembly import (
    DEFAULT_MAH_PARAMS as DEFAULT_MAH_PARAM_DICT,
)
from diffmah.individual_halo_assembly import _calc_halo_history

from ..stars import DEFAULT_SFR_PARAMS, _SFR_PARAM_BOUNDS
from ..stars import _sfr_history_from_mah, LGT0
from ..stars import _get_unbounded_sfr_params, _get_bounded_sfr_params
from ..quenching import _get_unbounded_q_params, _get_bounded_q_params
from ..utils import _get_dt_array
from ..quenching import DEFAULT_Q_PARAMS as DEFAULT_Q_U_PARAM_DICT
from ..stars import DEFAULT_SFR_PARAMS as DEFAULT_MS_U_PARAM_DICT

DEFAULT_MS_PARAMS = jnp.array(
    _get_bounded_sfr_params(*tuple(DEFAULT_MS_U_PARAM_DICT.values()))
)
DEFAULT_Q_PARAMS = jnp.array(
    _get_bounded_q_params(*tuple(DEFAULT_Q_U_PARAM_DICT.values()))
)

DEFAULT_MAH_PARAMS = jnp.array((12.0, 0.25, 2.0, 1.0))


def _get_default_mah_params():
    """Return (logt0, logmp, logtc, k, early, late)"""
    k = DEFAULT_MAH_PARAM_DICT["mah_k"]
    logmp = DEFAULT_MAH_PARAMS[0]
    logtc = DEFAULT_MAH_PARAMS[1]
    early_index = DEFAULT_MAH_PARAMS[2]
    late_index = DEFAULT_MAH_PARAMS[3]
    all_mah_params = [LGT0, logmp, logtc, k, early_index, late_index]
    return jnp.array(all_mah_params)


def test_sfh_parameter_bounds():
    for key, val in DEFAULT_SFR_PARAMS.items():
        assert _SFR_PARAM_BOUNDS[key][0] < val < _SFR_PARAM_BOUNDS[key][1]


def calc_sfh_on_default_params():
    mah_params = _get_default_mah_params()
    n_t = 100
    lgt = jnp.linspace(-1, LGT0, n_t)
    dt = _get_dt_array(10 * lgt)
    dmhdt, log_mah = _calc_halo_history(lgt, *mah_params)
    u_ms_params = jnp.array(_get_unbounded_sfr_params(*DEFAULT_MS_PARAMS))
    u_q_params = jnp.array(_get_unbounded_q_params(*DEFAULT_Q_PARAMS))
    args = lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params
    sfh = _sfr_history_from_mah(*args)
    return args, sfh


def test_sfh_on_default_params_does_not_change():
    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args
    n_t = dt.size
    assert sfh.shape == (n_t,)
