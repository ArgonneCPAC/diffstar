"""
"""
import numpy as np
from jax import jit as jjit
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt
from scipy.optimize import minimize


def minimizer(loss_func, loss_func_deriv, p_init, loss_data, nstep, *args, **kwargs):
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
    n_step : int
        Number of steps to walk down the gradient. Only used by Adam.

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

    if not res.success:
        _res = jax_adam_wrapper_v2(
            loss_func, p_init, loss_data, nstep, n_warmup=2, *args, **kwargs
        )
        p_best, loss_best, loss_arr, params_arr, fit_terminates = _res
        success = 0 if fit_terminates else -1

    return p_best, loss_best, success


def return_random_pinit(p_init, loss_data, loss_func_deriv):
    """Slightly perturb the initial guess with a Gaussian perturbation.

    Makes sure that the gradient does not contain NaNs.

    """

    p_init_2 = np.random.normal(p_init[0], p_init[1])

    isnan = np.isnan(loss_func_deriv(p_init_2, loss_data)).any()

    if isnan:
        i = 0
        while (i < 1000) & (isnan):
            p_init_2 = np.random.normal(p_init[0], p_init[1])
            isnan = np.isnan(loss_func_deriv(p_init_2, loss_data)).any()
            i += 1
        return p_init_2, isnan

    else:
        return p_init_2, isnan


def minimizer_wrapper(
    loss_func,
    loss_func_deriv,
    p_init,
    loss_data,
    loss_tol=0.1,
    max_iter=10,
):
    """Convenience function wrapping scipy's L-BFGS-B optimizer

    Starting from p_init, L-BFGS-B goes down the gradient
    to calculate the returned value p_best.

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
    loss_tol : float
        When loss_best < loss_tol the fitter stops. Otherwise it starts from
        a slightly different intial guess.
    max_iter : int
        While loss_best > loss_tol the fitter will restart from a slightly
        different initial guess, but it will stop after max_iter times.

    Returns
    -------
    p_best : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps
    loss_best : float
        Final value of the loss
    success : int
        -1 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    iter_id = 0

    p_best_list = []
    loss_best_list = []
    success_list = []

    loss_best_current = np.inf

    while (iter_id < max_iter) & (loss_best_current > loss_tol):
        p_init_run, isnan = return_random_pinit(p_init, loss_data, loss_func_deriv)
        if isnan:
            p_best_list.append(p_init[0])
            loss_best_list.append(999.99)
            success_list.append(-1)

            loss_best_current = np.min(loss_best_list)
            iter_id += 1
        else:
            res = minimize(
                loss_func,
                x0=p_init_run,
                method="L-BFGS-B",
                jac=loss_func_deriv,
                args=(loss_data,),
            )
            p_best = res.x
            loss_best = float(res.fun)
            success = 1

            p_best_list.append(p_best)
            loss_best_list.append(loss_best)
            success_list.append(success)

            if np.isnan(loss_best_list).all():
                loss_best_current = np.inf
            else:
                loss_best_current = np.nanmin(loss_best_list)

            iter_id += 1

    if np.isnan(loss_best_list).all():
        loss_best = 999.99
        success = -1
        p_best = p_init[0]
    else:
        argmin = np.nanargmin(loss_best_list)
        p_best = p_best_list[argmin]
        loss_best = loss_best_list[argmin]
        success = success_list[argmin]

    return p_best, loss_best, success


def jax_adam_wrapper_v2(
    loss_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
):
    """Convenience function wrapping JAX's Adam optimizer

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.
        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.
    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters
    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)
    n_step : int
        Number of steps to walk down the gradient
    n_warmup : int, optional
        Number of warmup iterations. At the end of the warmup, the best-fit parameters
        are used as input parameters to the final burn. Default is zero.
    warmup_n_step : int, optional
        Number of Adam steps to take during warmup. Default is 50.
    warmup_step_size : float, optional
        Step size to use during warmup phase. Default is 5*step_size.
    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01.

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps
    loss : float
        Final value of the loss
    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step
    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step
    fit_terminates : int
        0 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    p_init = np.copy(params_init)
    for i in range(n_warmup):
        p_init = _jax_adam_wrapper(
            loss_func, p_init, loss_data, warmup_n_step, step_size=warmup_step_size
        )[0]

    if np.all(np.isfinite(p_init)):
        p0 = p_init
    else:
        p0 = params_init

    _res = _jax_adam_wrapper(loss_func, p0, loss_data, n_step, step_size=step_size)
    if len(_res[2]) < n_step:
        fit_terminates = 0
    else:
        fit_terminates = 1
    return (*_res, fit_terminates)


def _jax_adam_wrapper(loss_func, params_init, loss_data, n_step, step_size=0.01):
    """Convenience function wrapping JAX's Adam optimizer

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.
        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.
    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters
    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)
    n_step : int
        Number of steps to walk down the gradient
    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps
    loss : float
        Final value of the loss
    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step
    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    """
    loss_arr = np.zeros(n_step).astype("f4") - 1.0
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    loss_vg = jjit(value_and_grad(loss_func, argnums=0))
    for istep in range(n_step):
        p = np.array(get_params(opt_state))

        loss, grads = loss_vg(p, loss_data)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
            if istep > 0:
                indx_best = np.nanargmin(loss_arr[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
            return (
                best_fit_params,
                best_fit_loss,
                loss_arr[:istep],
                params_arr[:istep, :],
            )
        else:
            params_arr[istep, :] = p
            loss_arr[istep] = loss
            opt_state = opt_update(istep, grads, opt_state)

    indx_best = np.nanargmin(loss_arr)
    best_fit_params = params_arr[indx_best]
    loss = loss_arr[indx_best]

    return best_fit_params, loss, loss_arr, params_arr
