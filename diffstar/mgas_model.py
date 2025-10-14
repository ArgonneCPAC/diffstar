from collections import namedtuple

from diffmah.diffmah_kernels import _log_mah_kern, mah_halopop, mah_singlehalo
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .defaults import FB, LGT0, T_TABLE_MIN
from .kernels.history_kernel_builders import _sfh_galpop_kern, _sfh_singlegal_kern
from .utils import _jax_get_dt_array, cumulative_mstar_formed

_cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))
_jax_get_dt_array_vmap = jjit(vmap(_jax_get_dt_array, in_axes=(0)))

GalHistory = namedtuple("GalHistory", ("sfh", "smh", "dmgash", "mgash"))

N_INT_STEPS = 20


@jjit
def calc_mgas_singlegal(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    """Calculate the Diffstar SFH and Mgas for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 8
        sfh_params is a tuple of floats
        lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, lg_qt, qlglgdt, lg_drop, lg_rejuv

    mah_params : namedtuple, length 4
        mah_params is a tuple of floats
        DiffmahParams = logmp, logtc, early_index, late_index

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


    Returns
    -------
    GalHistory namedtuple with 4 elements:

        sfh : ndarray, shape (nt, )
            Star formation rate in units of Msun/yr

        smh : ndarray, shape (nt, )
            Stellar mass in units of Msun

        dmgas_dt : ndarray, shape (nt, )
            Gas accretion rate - star formation rate, in units of Msun/yr

        mgas : ndarray, shape (nt, )
            Total remaining gas mass in units of Msun

    """
    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    args = (tarr, mah_params, ms_params, q_params, lgt0, fb)
    sfh = _sfh_singlegal_kern(*args)

    dmhdt, log_mah = mah_singlehalo(mah_params, tarr, lgt0)

    dmgasdt_inst = fb * dmhdt
    smh = cumulative_mstar_formed(tarr, sfh)
    mgas_inst = cumulative_mstar_formed(tarr, dmgasdt_inst)
    mgas = mgas_inst - smh

    dt = _jax_get_dt_array(tarr)
    dmgas_dt = _jax_get_dt_array(mgas) / dt / 1e9

    return GalHistory(sfh, smh, dmgas_dt, mgas)


@jjit
def calc_mgas_singlegal2(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    log_mah = _log_mah_kern(mah_params, tarr, lgt0)
    mgas_inst = fb * 10**log_mah

    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    sfh = _sfh_singlegal_kern(tarr, mah_params, ms_params, q_params, lgt0, fb)
    smh = cumulative_mstar_formed(tarr, sfh)

    mgas = mgas_inst - smh

    dt = _jax_get_dt_array(tarr)
    dmgas_dt = _jax_get_dt_array(mgas) / dt / 1e9

    return GalHistory(sfh, smh, dmgas_dt, mgas)


@jjit
def _calc_mgas_kern(sfh_params, mah_params, t_obs, lgt0, fb):

    t_table = jnp.linspace(T_TABLE_MIN, t_obs, N_INT_STEPS)
    log_mah_table = _log_mah_kern(mah_params, t_table, lgt0)
    mgas_inst_table = fb * 10**log_mah_table

    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    sfh_table = _sfh_singlegal_kern(t_table, mah_params, ms_params, q_params, lgt0, fb)
    smh_table = cumulative_mstar_formed(t_table, sfh_table)

    mgas_table = mgas_inst_table - smh_table

    mgas_obs = mgas_table[-1]
    sfr_obs = sfh_table[-1]
    mstar_obs = smh_table[-1]

    return mgas_obs, mstar_obs, sfr_obs


@jjit
def _calc_dmgas_dt_kern_wrapper(sfh_params, mah_params, t_obs, lgt0, fb):
    mgas_obs = _calc_mgas_kern(sfh_params, mah_params, t_obs, lgt0, fb)[0]
    return mgas_obs


_calc_dmgas_dt_kern_nonorm = jjit(grad(_calc_dmgas_dt_kern_wrapper, argnums=2))


@jjit
def _calc_mgas_and_dmgas_dt_kern(sfh_params, mah_params, t_obs, lgt0, fb):
    mgas_obs, mstar_obs, sfr_obs = _calc_mgas_kern(
        sfh_params, mah_params, t_obs, lgt0, fb
    )
    dmgas_dt = _calc_dmgas_dt_kern_nonorm(sfh_params, mah_params, t_obs, lgt0, fb) / 1e9
    return sfr_obs, mstar_obs, dmgas_dt, mgas_obs


@jjit
def _calc_dmgas_dt_kern(sfh_params, mah_params, t_obs, lgt0, fb):
    return _calc_dmgas_dt_kern_nonorm(sfh_params, mah_params, t_obs, lgt0, fb) / 1e9


_TARR = (None, None, 0, None, None)
_calc_mgas_and_dmgas_dt_vmap = jjit(vmap(_calc_mgas_and_dmgas_dt_kern, in_axes=_TARR))


@jjit
def calc_mgas_singlegal3(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    _res = _calc_mgas_and_dmgas_dt_vmap(sfh_params, mah_params, tarr, lgt0, fb)
    sfh, smh, dmgas_dt, mgash = _res
    return GalHistory(sfh, smh, dmgas_dt, mgash)


_GPOP = (0, 0, None, None, None)
_calc_mgas_and_dmgas_dt_galpop = jjit(vmap(_calc_mgas_and_dmgas_dt_vmap, in_axes=_GPOP))


@jjit
def calc_mgas_galpop3(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    _res = _calc_mgas_and_dmgas_dt_galpop(sfh_params, mah_params, tarr, lgt0, fb)
    sfh, smh, dmgas_dt, mgash = _res
    return GalHistory(sfh, smh, dmgas_dt, mgash)


@jjit
def calc_mgas_galpop(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    """Calculate the Diffstar SFH and Mgas for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 8
        sfh_params is a tuple of ndarrays of shape (ngals, )
        lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, lg_qt, qlglgdt, lg_drop, lg_rejuv

    mah_params : namedtuple, length 4
        mah_params is a tuple of ndarrays of shape (ngals, )
        DiffmahParams = logmp, logtc, early_index, late_index

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

    Returns
    -------
    GalHistory namedtuple with 4 elements:

        sfh : ndarray, shape (nt, )
            Star formation rate in units of Msun/yr

        smh : ndarray, shape (nt, )
            Stellar mass in units of Msun

        dmgas_dt : ndarray, shape (nt, )
            Gas accretion rate - star formation rate, in units of Msun/yr

        mgas : ndarray, shape (nt, )
            Total remaining gas mass in units of Msun

    """
    ms_params, q_params = sfh_params[:4], sfh_params[4:]
    args = (tarr, mah_params, ms_params, q_params, lgt0, fb)
    sfh = _sfh_galpop_kern(*args)

    dmhdt, log_mah = mah_halopop(mah_params, tarr, lgt0)

    dmgasdt_inst = fb * dmhdt
    smh = _cumulative_mstar_formed_vmap(tarr, sfh)
    mgas_inst = _cumulative_mstar_formed_vmap(tarr, dmgasdt_inst)
    mgas = mgas_inst - smh

    dt = _jax_get_dt_array(tarr)
    dmgas_dt = _jax_get_dt_array_vmap(mgas) / dt / 1e9

    return GalHistory(sfh, smh, dmgas_dt, mgas)
