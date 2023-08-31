"""
"""
import numpy as np
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp


@jjit
def _jax_get_dt_array(t):
    dt = jnp.zeros_like(t)
    tmids = 0.5 * (t[:-1] + t[1:])
    dtmids = jnp.diff(tmids)

    dt = dt.at[1:-1].set(dtmids)

    t_lo = t[0] - (t[1] - t[0]) / 2
    t_hi = t[-1] + dtmids[-1] / 2

    dt = dt.at[0].set(tmids[0] - t_lo)
    dt = dt.at[-1].set(t_hi - tmids[-1])
    return dt


def _get_dt_array(t):
    """Compute delta time from input time.

    Parameters
    ----------
    t : ndarray of shape (n, )

    Returns
    -------
    dt : ndarray of shape (n, )

    Returned dt is defined by time interval (t_lo, t_hi),
    where t_lo^i = 0.5(t_i-1 + t_i) and t_hi^i = 0.5(t_i + t_i+1)

    """
    n = t.size
    dt = np.zeros(n)
    tlo = t[0] - (t[1] - t[0]) / 2
    for i in range(n - 1):
        thi = (t[i + 1] + t[i]) / 2
        dt[i] = thi - tlo
        tlo = thi
    thi = t[n - 1] + dt[n - 2] / 2
    dt[n - 1] = thi - tlo
    return dt


def get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    """Sigmoid function implemented w/ `jax.numpy.exp`

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ymin : float or array-like
        The value as x goes to -infty.
    ymax : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input

    """
    height_diff = ymax - ymin
    return ymin + height_diff * lax.logistic(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def sigmoid_poly(x, x0, k, ymin, ymax):
    arg = k * (x - x0)
    body = 0.5 * arg / jnp.sqrt(1 + arg**2) + 0.5
    return ymin + (ymax - ymin) * body


@jjit
def tw_cuml_jax_kern(x, m, h):
    """CDF of the triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF.

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 1.0,
            lambda xx: (
                -5 * xx**7 / 69984
                + 7 * xx**5 / 2592
                - 35 * xx**3 / 864
                + 35 * xx / 96
                + 1 / 2
            ),
            x,
        ),
        y,
    )


@jjit
def tw_bin_jax_kern(m, h, L, H):
    """Integrated bin weight for the triweight kernel.

    Parameters
    ----------
    m : array-like or scalar
        The value at which to evaluate the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.
    L : array-like or scalar
        The lower bin limit.
    H : array-like or scalar
        The upper bin limit.

    Returns
    -------
    bin_prob : array-like or scalar
        The value of the kernel integrated over the bin.

    """
    return tw_cuml_jax_kern(H, m, h) - tw_cuml_jax_kern(L, m, h)


@jjit
def jax_np_interp(x, xt, yt, indx_hi):
    """JAX-friendly implementation of np.interp.
    Requires indx_hi to be precomputed, e.g., using np.searchsorted.

    Parameters
    ----------
    x : ndarray of shape (n, )
        Abscissa values in the interpolation
    xt : ndarray of shape (k, )
        Lookup table for the abscissa
    yt : ndarray of shape (k, )
        Lookup table for the ordinates

    Returns
    -------
    y : ndarray of shape (n, )
        Result of linear interpolation

    """
    indx_lo = indx_hi - 1
    xt_lo = xt[indx_lo]
    xt_hi = xt[indx_hi]
    dx_tot = xt_hi - xt_lo
    yt_lo = yt[indx_lo]
    yt_hi = yt[indx_hi]
    dy_tot = yt_hi - yt_lo
    m = dy_tot / dx_tot
    y = yt_lo + m * (x - xt_lo)
    return y
