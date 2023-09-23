"""Functions ms_param_clipper and q_param_clipper implement clips on the diffstar
parameters to help protect against NaNs and infinities
"""
from collections import OrderedDict

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..kernels.main_sequence_kernels import MS_PARAM_BOUNDS_PDICT
from ..kernels.quenching_kernels import Q_PARAM_BOUNDS_PDICT

_EPS = 0.001

MS_CLIPPING_DICT = OrderedDict()
Q_CLIPPING_DICT = OrderedDict()

for key, bounds in MS_PARAM_BOUNDS_PDICT.items():
    lo, hi = bounds
    MS_CLIPPING_DICT[key] = (lo + _EPS, hi - _EPS)
for key, bounds in Q_PARAM_BOUNDS_PDICT.items():
    lo, hi = bounds
    Q_CLIPPING_DICT[key] = (lo + _EPS, hi - _EPS)

MS_CLIPS_LO = jnp.array([x[0] for x in MS_CLIPPING_DICT.values()])
MS_CLIPS_HI = jnp.array([x[1] for x in MS_CLIPPING_DICT.values()])
Q_CLIPS_LO = jnp.array([x[0] for x in Q_CLIPPING_DICT.values()])
Q_CLIPS_HI = jnp.array([x[1] for x in Q_CLIPPING_DICT.values()])


@jjit
def _clipping_kern(arr, lo, hi):
    msk_lo = arr <= lo
    msk_hi = arr >= hi
    arr = jnp.where(msk_lo, lo, arr)
    arr = jnp.where(msk_hi, hi, arr)
    return arr


_clipping_kern_vmap = jjit(vmap(_clipping_kern, in_axes=(0, 0, 0)))


@jjit
def ms_param_clipper(ms_params):
    """Clip the main sequence parameters to be at least epsilon away from the bounds

    Parameters
    ----------
    ms_params : ndarray, shape (n, 5)

    Returns
    ----------
    clipped_ms_params : ndarray, shape (n, 5)

    """
    clipped_ms_params = _clipping_kern_vmap(ms_params.T, MS_CLIPS_LO, MS_CLIPS_HI).T
    return clipped_ms_params


@jjit
def q_param_clipper(q_params):
    """Clip the quenching parameters to be at least epsilon away from the bounds

    Parameters
    ----------
    q_params : ndarray, shape (n, 4)

    Returns
    ----------
    clipped_q_params : ndarray, shape (n, 4)

    """
    clipped_q_params = _clipping_kern_vmap(q_params.T, Q_CLIPS_LO, Q_CLIPS_HI).T
    rejuv_clip = clipped_q_params[:, 2] + _EPS / 2
    msk_rejuv = clipped_q_params[:, 3] <= rejuv_clip
    new_lg_rejuv = jnp.where(msk_rejuv, rejuv_clip, clipped_q_params[:, 3])
    clipped_q_params = clipped_q_params.at[:, 3].set(new_lg_rejuv)
    return clipped_q_params
