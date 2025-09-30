""" """

from jax import jit as jjit
from jax import numpy as jnp

from ..defaults import DEFAULT_DIFFSTAR_PARAMS
from .defaults import DEFAULT_DIFFSTARPOP_U_PARAMS


@jjit
def get_all_params_from_varied(varied, defaults):
    """Replace entries of namedtuple defaults with subset found in varied

    Params
    ------
    varied : namedtuple
        Each entry of varied._fields should also appear in defaults._fields

    defaults : namedtuple
        Default values of all parameters in the model

    Returns
    -------
    params : namedtuple
        Same as defaults except for fields appearing in varied

    """
    return defaults._replace(**varied._asdict())


@jjit
def get_all_diffstarpop_u_params(varied_u_params):
    """Get the full collection of diffstarpop unbounded params from some subset

    Parameters
    ----------
    varied_u_params : namedtuple
        varied_u_params is a namedtuple with the same 5 entries as DiffstarPopUParams
            u_sfh_pdf_cens_params
            u_satquench_params
        Each entry is itself a namedtuple of parameters, which can be any subset
        of the same parameters appearing in that component of DiffstarPopUParams

    Returns
    -------
    u_params : namedtuple
        Instance of DiffstarPopUParams. Values will be taken from varied_u_params when
        present, and otherwise will be taken from DEFAULT_DIFFSTARPOP_U_PARAMS

    """
    _diffstarpop_u_params = get_all_params_from_varied(
        varied_u_params, DEFAULT_DIFFSTARPOP_U_PARAMS
    )
    return DEFAULT_DIFFSTARPOP_U_PARAMS._make(_diffstarpop_u_params)


def mc_select_diffstar_params(sfh_params_1, sfh_params_0, mc_is_1):
    """Select Monte Carlo realization of diffstar params

    Parameters
    ----------
    sfh_params_1 : namedtuple of sfh_params
        sfh_params_1.ms_params stores main sequence params, 4 arrays with shape (n, )
        sfh_params_1.q_params stores quenching params, 4 arrays with shape (n, )

    sfh_params_0 : namedtuple of sfh_params
        sfh_params_0.ms_params stores main sequence params, 4 arrays with shape (n, )
        sfh_params_0.q_params stores quenching params, 4 arrays with shape (n, )

    mc_is_1 : bool
        Boolean array, shape (n, )
        Equals 1 for sfh_params1 and 0 for sfh_params0

    Returns
    -------
    sfh_params: namedtuple of sfh_params
        sfh_params.ms_params stores main sequence params, 4 arrays with shape (n, )
        sfh_params.q_params stores quenching params, 4 arrays with shape (n, )

    """
    sfh_params = [
        jnp.where(mc_is_1, getattr(sfh_params_1, x), getattr(sfh_params_0, x))
        for x in DEFAULT_DIFFSTAR_PARAMS._fields
    ]
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(sfh_params)
    return sfh_params
