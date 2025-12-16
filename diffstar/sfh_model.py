""" """

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .kernels.history_kernel_builders import _sfh_galpop_kern, _sfh_singlegal_kern
from .utils import cumulative_mstar_formed

_cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))

GalHistory = namedtuple("GalHistory", ("sfh", "smh"))


@partial(jjit, static_argnames="return_smh")
def calc_sfh_singlegal(
    sfh_params,
    mah_params,
    tarr,
    *,
    lgt0,
    fb,
    return_smh=False,
):
    """Calculate the Diffstar SFH for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 8
        sfh_params is a tuple of floats
        lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, lg_qt, qlglgdt, lg_drop, lg_rejuv

    mah_params : namedtuple, length 5
        mah_params is a tuple of floats
        DiffmahParams = logmp, logtc, early_index, late_index, t_peak

    t_peak : float

    tarr : ndarray, shape (nt, )

    lgt0 : float, optional
        Base-10 log of the z=0 age of the Universe in Gyr
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    fb : float, optional
        Cosmic baryon fraction Ob0/Om0
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    return_smh : bool, optional
        If True, function return sfh, smh,
        where smh is the history of stellar mass formed in units of Msun
        Default is False, in which case function only returns sfh

    Returns
    -------
    sfh : ndarray, shape (nt, )
        Star formation rate in units of Msun/yr

    smh : ndarray, shape (nt, ), optional
        Stellar mass in units of Msun
        This variable is only returned if return_smh=True

    """
    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    args = (tarr, mah_params, ms_params, q_params, lgt0, fb)
    sfh = _sfh_singlegal_kern(*args)
    if return_smh:
        smh = cumulative_mstar_formed(tarr, sfh)
        return GalHistory(sfh, smh)
    else:
        return sfh


@partial(jjit, static_argnames="return_smh")
def calc_sfh_galpop(
    sfh_params,
    mah_params,
    tarr,
    *,
    lgt0,
    fb,
    return_smh=False,
):
    """Calculate the Diffstar SFH for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 8
        sfh_params is a tuple of ndarrays of shape (ngals, )
        lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, lg_qt, qlglgdt, lg_drop, lg_rejuv

    mah_params : namedtuple, length 5
        mah_params is a tuple of ndarrays of shape (ngals, )
        DiffmahParams = logmp, logtc, early_index, late_index, t_peak

    t_peak : ndarray, shape (ngals, )

    tarr : ndarray, shape (nt, )

    lgt0 : float, optional
        Base-10 log of the z=0 age of the Universe in Gyr
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    fb : float, optional
        Cosmic baryon fraction Ob0/Om0
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    return_smh : bool, optional
        If True, function return sfh, smh,
        where smh is the history of stellar mass formed in units of Msun
        Default is False, in which case function only returns sfh

    Returns
    -------
    sfh : ndarray, shape (ngals, nt)
        Star formation rate in units of Msun/yr

    smh : ndarray, shape (ngals, nt), optional
        Stellar mass in units of Msun
        This variable is only returned if return_smh=True

    """
    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    args = (
        tarr,
        jnp.array(mah_params).T,
        jnp.array(ms_params).T,
        jnp.array(q_params).T,
        lgt0,
        fb,
    )
    sfh = _sfh_galpop_kern(*args)
    if return_smh:
        smh = _cumulative_mstar_formed_vmap(tarr, sfh)
        return GalHistory(sfh, smh)
    else:
        return sfh
