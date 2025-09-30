""" """

from collections import namedtuple
from copy import deepcopy

import numpy as np

from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS

from ..defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from ..param_utils import get_all_diffstarpop_u_params, mc_select_diffstar_params


def test_get_all_ms_massonly_params_from_varied():
    u_sfh_pdf_cens_pdict = deepcopy(
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params._asdict()
    )
    for key, val in u_sfh_pdf_cens_pdict.items():
        if key.startswith("u_mean_"):
            u_sfh_pdf_cens_pdict[key] = val + 0.1

    Params = namedtuple("Params", u_sfh_pdf_cens_pdict.keys())
    u_sfh_pdf_cens_params = Params(**u_sfh_pdf_cens_pdict)

    varied_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._replace(
        u_sfh_pdf_cens_params=u_sfh_pdf_cens_params
    )
    all_u_params = get_all_diffstarpop_u_params(varied_u_params)

    u_sfh_pdf_cens_pdict_default = deepcopy(
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params._asdict()
    )
    for key, val in u_sfh_pdf_cens_pdict_default.items():
        if key.startswith("u_mean_"):
            assert getattr(all_u_params.u_sfh_pdf_cens_params, key) == val + 0.1
        else:
            assert getattr(all_u_params.u_sfh_pdf_cens_params, key) == val


def test_mc_select_diffstar_params():
    mc_is_1 = np.concatenate((np.ones(5), np.zeros(5)))
    ngals = mc_is_1.size
    ZZ1 = np.zeros(ngals) - 0.1
    ZZ2 = np.zeros(ngals) + 0.1

    sfh_params_1 = DEFAULT_DIFFSTAR_PARAMS._make(
        [
            getattr(DEFAULT_DIFFSTAR_PARAMS, x) + ZZ1
            for x in DEFAULT_DIFFSTAR_PARAMS._fields
        ]
    )
    sfh_params_2 = DEFAULT_DIFFSTAR_PARAMS._make(
        [
            getattr(DEFAULT_DIFFSTAR_PARAMS, x) + ZZ2
            for x in DEFAULT_DIFFSTAR_PARAMS._fields
        ]
    )

    sfh_params = mc_select_diffstar_params(sfh_params_1, sfh_params_2, mc_is_1)

    for pname in sfh_params._fields:
        val = getattr(sfh_params, pname)
        val1 = getattr(sfh_params_1, pname)
        val2 = getattr(sfh_params_2, pname)
        assert np.allclose(val[:5], val1[:5])
        assert np.allclose(val[5:], val2[5:])
