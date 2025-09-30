""" """

from jax import jit as jjit
from jax import lax, nn
from jax import numpy as jnp
from jax import vmap
from scipy.optimize import minimize

from ..utils import jax_np_interp


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
    return ymin + height_diff * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


def minimizer(loss_func, loss_func_deriv, p_init, loss_data):
    """Minimizer mixing scipy's L-BFGS-B minimizer with JAX's Adam as a backup plan.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.
        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.
    loss_func_deriv : callable
        Returns the gradient wrt the parameters of loss_func.
        Must accept inputs (params, data) and return a scalar.
    p_init : ndarray of shape (n_params, )
        Initial guess at the parameters. The fitter uses this guess to draw
        random initial guesses with small perturbations around these values.
    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(p_init, loss_data)

    Returns
    -------
    p_best : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps
    loss_best : float
        Final value of the loss
    success : int
        -1 if NaN or inf is encountered by the fitter, causing termination before n_step
        0 for a fit that fails with L-BFGS-B but terminates without problems using Adam
        1 for a fit that terminates with no such problems using L-BFGS-B

    """

    res = minimize(
        loss_func, x0=p_init, method="L-BFGS-B", jac=loss_func_deriv, args=(loss_data,)
    )
    p_best = res.x
    loss_best = float(res.fun)
    success = 1

    return p_best, loss_best, success


@jjit
def _tw_cuml_lax_kern(x, m, h):
    """Triweight kernel version of an err function.
    This kernel accepts and returns scalars for all arguments
    """
    z = (x - m) / h
    val = -5 * z**7 / 69984 + 7 * z**5 / 2592 - 35 * z**3 / 864 + 35 * z / 96 + 1 / 2
    val = lax.cond(z < -3, lambda s: 0.0, lambda s: val, z)
    val = lax.cond(z > 3, lambda s: 1.0, lambda s: val, z)
    return val


@jjit
def _tw_bin_weight_lax_kern(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin.
    This kernel accepts and returns scalars for all arguments
    """
    a = _tw_cuml_lax_kern(x, lo, sig)
    b = _tw_cuml_lax_kern(x, hi, sig)
    return a - b


jax_np_interp_vmap = jjit(vmap(jax_np_interp, in_axes=(0, 0, None, 0)))


@jjit
def correlation_from_covariance(cov):
    """Correlation matrix from covariance matrix

    Parameters
    ----------
    cov : array, shape (n, n)

    Returns
    -------
    corr : array, shape (n, n)

    """
    v = jnp.sqrt(jnp.diag(cov))
    outer_v = jnp.outer(v, v)
    corr = cov / outer_v
    msk = cov == 0
    corr = jnp.where(msk, 0.0, corr)
    return corr


@jjit
def covariance_from_correlation(corr, evals):
    """Covariance matrix from correlation matrix

    Parameters
    ----------
    corr : array, shape (n, n)

    evals : array, shape (n, )
        Array of eigenvalues, e.g. (σ_1, σ_2, ..., σ_n)
        Note that np.diag(cov) = evals**2

    Returns
    -------
    cov : array, shape (n, n)

    """
    D = jnp.diag(evals)
    cov = jnp.dot(jnp.dot(D, corr), D)
    return cov


@jjit
def smoothly_clipped_line(x, x0, y0, m, y_lo, y_hi):
    """
    Generates a smoothly clipped straight line with specified parameters.

    This function computes a line with an initial intercept `y0`, a slope `m`,
    and a zero point at `x0`. The line is smoothly clipped within the bounds
    `y_lo` and `y_hi` using a sigmoid-based transition to avoid sharp cutoffs.

    Parameters:
    x : array-like
        The x-axis values at which to evaluate the function.
    x0 : float
        The reference x-value where y = y0.
    y0 : float
        The initial y-intercept of the line at x = x0.
    m : float
        The slope of the line.
    y_lo : float
        The lower bound for the y-values.
    y_hi : float
        The upper bound for the y-values.

    Returns:
    y_clipped : array-like
        The y-values corresponding to x, smoothly clipped within [y_lo, y_hi].

    """
    x_lo = (y_lo - y0) / m + x0  # value of x at which y=y_lo
    x_hi = (y_hi - y0) / m + x0  # value of x at which y=y_hi

    dx_lo = x0 - x_lo
    dx_hi = x_hi - x0

    eps_lo = dx_lo / 100.0  # tiny step in x above x_lo
    eps_hi = dx_hi / 100.0  # tiny step in x below x_hi

    xc_lo = x_lo + eps_lo  # value of x at which y=y_lo+ε
    xc_hi = x_hi - eps_hi  # value of x at which y=y_hi-ε

    y_lo_bound = y0 + m * (xc_lo - x0)  # value of y=y_lo+ε
    y_hi_bound = y0 + m * (xc_hi - x0)  # value of y=y_hi-ε

    CLIPPING_K = 20.0 / eps_lo  # Steep transition speed for this Δx
    y_unclipped = y0 + m * (x - x0)
    y_clipped_from_below = _sigmoid(x, xc_lo, CLIPPING_K, y_lo_bound, y_unclipped)
    y_clipped = _sigmoid(x, xc_hi, CLIPPING_K, y_clipped_from_below, y_hi_bound)
    return y_clipped
