"""Alternative implementation of _gas_conversion_kern based on grad"""
from jax import jit as jjit
from jax import lax
from jax import grad


@jjit
def unit_tw_cuml_jax_kern(z):
    """CDF of the triweight kernel.

    Rises from 0 at z=-3 to unity at z=3.

    Parameters
    ----------
    z : float
        z-score

    Returns
    -------
    cdf : float
        The value of the kernel CDF.

    """
    return lax.cond(
        z < -3,
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
        z,
    )


@jjit
def tw_cuml_jax_kern(x, m, w):
    """Alternate version of unit_tw_cuml_jax_kern with variable midpoint and width"""
    z = (x - m) / w
    return unit_tw_cuml_jax_kern(z)


@jjit
def _half_unit_tw_cuml_kern(z):
    """Alternate version of unit_tw_cuml_jax_kern: rises from 0 at z=0 to 1 at z=3"""
    half_unit = lax.cond(z < 0, lambda x: 0.5, lambda x: unit_tw_cuml_jax_kern(x), z)
    res = 2 * (half_unit - 0.5)
    return res


@jjit
def _half_unit_tw_cuml_kern_variable_m_w(x, m, w):
    """Alternate version of _half_unit_tw_cuml_kern with variable midpoint and width"""
    z = (x - m) / w
    return _half_unit_tw_cuml_kern(z)


_tw_dmgas_dt_kern = jjit(grad(_half_unit_tw_cuml_kern_variable_m_w, argnums=0))
