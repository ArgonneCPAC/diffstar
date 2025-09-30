from collections import OrderedDict

import jax
import numpy as np
from jax import numpy as jnp

from ..kernels.defaults_mgash import (
    DEFAULT_DIFFSTARPOP_U_PARAMS as DEFAULT_DIFFSTARPOP_U_PARAMS_tpeak,  # DEFAULT_DIFFSTARPOP_PARAMS,
)


def register_tuple_new_diffstarpop_tpeak(named_tuple_class):
    jax.tree_util.register_pytree_node(
        named_tuple_class,
        # tell JAX how to unpack the NamedTuple to an iterable
        lambda x: (tuple_to_jax_array(x), None),
        # tell JAX how to pack it back into the proper NamedTuple structure
        lambda _, x: array_to_tuple_new_diffstarpop_tpeak(x, named_tuple_class),
    )


def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def tuple_to_jax_array(t):
    res = tuple(flatten_tuples(t))
    return jnp.asarray(res)


def tuple_to_array(t):
    res = tuple(flatten_tuples(t))
    return np.asarray(res)


def array_to_tuple_new_diffstarpop_tpeak(a, t):

    count = 0

    SFH_params = DEFAULT_DIFFSTARPOP_U_PARAMS_tpeak.u_sfh_pdf_cens_params
    new_count = count + len(SFH_params)
    new_sfh_pdf_cens_params_u_params = SFH_params._make(a[count:new_count])

    SAT_params = DEFAULT_DIFFSTARPOP_U_PARAMS_tpeak.u_satquench_params
    new_count2 = new_count + len(SAT_params)
    new_satquenchpop_u_params = SAT_params._make(a[new_count:new_count2])

    _up = (new_sfh_pdf_cens_params_u_params, new_satquenchpop_u_params)
    new_diffstarpop_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS_tpeak._make(_up)

    new_dict = OrderedDict(diffstarpop_u_params=new_diffstarpop_u_params)

    T = t(*list(new_dict.values()))

    return T
