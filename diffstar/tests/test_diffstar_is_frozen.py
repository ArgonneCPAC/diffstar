"""Unit tests enforcing that the behavior of Diffstar on the default params is frozen.
"""
import os
import numpy as np
from jax import numpy as jnp
from diffmah.individual_halo_assembly import DEFAULT_MAH_PARAMS
from diffmah.individual_halo_assembly import _get_early_late
from diffmah.individual_halo_assembly import _calc_halo_history

from ..stars import _sfr_history_from_mah, LGT0
from ..stars import _get_unbounded_sfr_params, _get_bounded_sfr_params
from ..stars import DEFAULT_SFR_PARAMS as DEFAULT_MS_U_PARAM_DICT
from ..quenching import DEFAULT_Q_PARAMS as DEFAULT_Q_U_PARAM_DICT
from ..quenching import _get_unbounded_q_params, _get_bounded_q_params
from ..utils import _get_dt_array

DEFAULT_MS_PARAMS = jnp.array(
    _get_bounded_sfr_params(*tuple(DEFAULT_MS_U_PARAM_DICT.values()))
)
DEFAULT_Q_PARAMS = jnp.array(
    _get_bounded_q_params(*tuple(DEFAULT_Q_U_PARAM_DICT.values()))
)

DEFAULT_LOGM0 = 12.0

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def _get_default_mah_params():
    """Return (logt0, logmp, logtc, k, early, late)"""
    mah_logtc, mah_k, mah_ue, mah_ul = list(DEFAULT_MAH_PARAMS.values())
    early_index, late_index = _get_early_late(mah_ue, mah_ul)
    k = DEFAULT_MAH_PARAMS["mah_k"]
    logmp = DEFAULT_LOGM0

    default_mah_params = [LGT0, logmp, mah_logtc, k, early_index, late_index]
    default_mah_params = jnp.array(default_mah_params)
    return default_mah_params


def calc_sfh_on_default_params():
    """Calculate SFH for the Diffstar and Diffmah default parameters.

    This function is used to generate the unit-testing data used in this module
    to freeze the behavior of Diffstar evaluated on the default parameters.
    """
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


def test_default_u_ms_params_do_not_change():
    """Enforce that the default parameter values of main sequence SFR are frozen.

    This unit test is used to enforce that the behavior of Diffstar SFH
    does not accidentally change as the code evolves. We may actually wish to update
    the default parameters of Diffstar in future, in which case this unit test will
    need to be updated. But generally speaking, changing this unit test should only
    be done deliberately and with a very good reason.

    """
    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args

    u_ms_fn = os.path.join(TESTING_DATA_DRN, "default_params_test_u_ms_params.txt")
    frozen_u_ms_params = np.loadtxt(u_ms_fn)
    assert np.allclose(u_ms_params, frozen_u_ms_params)


def test_default_u_q_params_do_not_change():
    """Enforce that the default parameter values of the quenching function are frozen.

    This unit test is used to enforce that the behavior of Diffstar SFH
    does not accidentally change as the code evolves. We may actually wish to update
    the default parameters of Diffstar in future, in which case this unit test will
    need to be updated. But generally speaking, changing this unit test should only
    be done deliberately and with a very good reason.

    """
    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args

    u_q_fn = os.path.join(TESTING_DATA_DRN, "default_params_test_u_q_params.txt")
    frozen_u_q_params = np.loadtxt(u_q_fn)
    assert np.allclose(u_q_params, frozen_u_q_params)


def test_sfh_on_default_params_is_frozen():
    """Enforce that the Diffstar SFH is frozen when evaluated on the default parameters.

    This unit test is used to enforce that the behavior of Diffstar SFH
    does not accidentally change as the code evolves. We may actually wish to update
    the default parameters of Diffstar in future, in which case this unit test will
    need to be updated. But generally speaking, changing this unit test should only
    be done deliberately and with a very good reason.

    """
    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args
    n_t = dt.size
    assert sfh.shape == (n_t,)

    sfh_fn = os.path.join(TESTING_DATA_DRN, "default_params_test_sfh.txt")
    frozen_sfh = np.loadtxt(sfh_fn)
    assert np.allclose(sfh, frozen_sfh, rtol=1e-4)


def test_diffmah_behavior_is_frozen():
    """Enforce that the Diffmah MAH assumed by Diffstar does not change.

    The Diffstar project publicly releases best-fitting approximations to external
    simulation data, e.g., by UniverseMachine and IllustrisTNG. These fits make
    some hard assumptions about the behavior of Diffmah. This unit test enforces that
    all such assumptions continue to hold. It may be that in future, the Diffmah code
    evolves in a harmless way that poses no problem for Diffstar; if that happens,
    this unit test guarantees that we will find out about any such change
    (at which point this test will need to be updated).

    """
    assumed_default_params = _get_default_mah_params()
    lgt0, logmp, logtc, k, early_index, late_index = assumed_default_params

    msg = "Default age of the universe assumed by Diffmah has changed"
    assert lgt0 == LGT0, msg

    msg = "Default logmp used in this testing module has changed"
    assert logmp == DEFAULT_LOGM0, msg

    msg = "Default mah_k parameter within Diffmah has changed"
    assert DEFAULT_MAH_PARAMS["mah_k"] == 3.5, msg

    msg = "mah_k parameter returned by _get_default_mah_params function has changed"
    assert k == 3.5, msg

    mah_params_fn = os.path.join(
        TESTING_DATA_DRN, "default_params_test_diffmah_params.txt"
    )
    frozen_diffmah_params = np.loadtxt(mah_params_fn)
    msg = "Default Diffmah parameters have changed"
    assert np.allclose(assumed_default_params, frozen_diffmah_params, rtol=1e-4), msg

    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args
    dmhdt, log_mah = _calc_halo_history(lgt, *assumed_default_params)

    log_mah_fn = os.path.join(TESTING_DATA_DRN, "default_params_test_log_mah.txt")
    frozen_log_mah = np.loadtxt(log_mah_fn)
    msg = "Diffmah log_mah has changed when evaluated on default parameters"
    assert np.allclose(frozen_log_mah, log_mah, rtol=1e-4), msg

    dmhdt_fn = os.path.join(TESTING_DATA_DRN, "default_params_test_dmhdt.txt")
    frozen_dmhdt = np.loadtxt(dmhdt_fn)
    msg = "Diffmah dmhdt has changed when evaluated on default parameters"
    assert np.allclose(frozen_dmhdt, dmhdt, rtol=1e-4), msg
