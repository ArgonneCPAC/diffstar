from collections import namedtuple

from diffmah.diffmah_kernels import mah_halopop, mah_singlehalo
from jax import jit as jjit
from jax import vmap

from .defaults_flat import FB, LGT0
from .kernels.history_kernel_builders_flat import _sfh_galpop_kern, _sfh_singlegal_kern
from .utils import _jax_get_dt_array, cumulative_mstar_formed

_cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))
_jax_get_dt_array_vmap = jjit(vmap(_jax_get_dt_array, in_axes=(0)))

GalHistory = namedtuple("GalHistory", ("sfh", "smh", "dmgash", "mgash"))


@jjit
def calc_mgas_singlegal(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    """Calculate the Diffstar SFH and Mgas for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 2
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv


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
def calc_mgas_galpop(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    """Calculate the Diffstar SFH and Mgas for a single galaxy

    Parameters
    ----------
    sfh_params : namedtuple, length 2
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv


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
    sfh = _sfh_galpop_kern(*args)

    dmhdt, log_mah = mah_halopop(mah_params, tarr, lgt0)

    dmgasdt_inst = fb * dmhdt
    smh = _cumulative_mstar_formed_vmap(tarr, sfh)
    mgas_inst = _cumulative_mstar_formed_vmap(tarr, dmgasdt_inst)
    mgas = mgas_inst - smh

    dt = _jax_get_dt_array(tarr)
    dmgas_dt = _jax_get_dt_array_vmap(mgas) / dt / 1e9

    return GalHistory(sfh, smh, dmgas_dt, mgas)
