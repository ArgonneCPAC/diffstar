"""
"""
import os

import numpy as np

from ...defaults import FB
from ...sfh import sfh_galpop
from ..main_sequence_kernels import (
    _get_bounded_sfr_params_vmap,
    _get_unbounded_sfr_params_vmap,
)
from ..quenching_kernels import _get_bounded_q_params_vmap, _get_unbounded_q_params_vmap

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_sfh_galpop_agrees_with_v0p1_bounded_params():
    t_table = np.loadtxt(os.path.join(TESTING_DATA_DRN, "t_table_testing_v0.1.0.txt"))
    lgt0 = np.log10(t_table[-1])
    mah_params = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "mah_params_testing_v0.1.0.txt")
    )
    ms_params = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "ms_params_testing_v0.1.0.txt")
    )
    q_params = np.loadtxt(os.path.join(TESTING_DATA_DRN, "q_params_testing_v0.1.0.txt"))
    sfh_table_v0p1 = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "sfh_table_testing_v0.1.0.txt")
    )
    sfh_table = sfh_galpop(
        t_table,
        mah_params,
        ms_params,
        q_params,
        lgt0=lgt0,
        fb=FB,
        ms_param_type="bounded",
        q_param_type="bounded",
    )
    assert np.allclose(sfh_table_v0p1, sfh_table, atol=0.02)
    assert np.allclose(sfh_table_v0p1, sfh_table, rtol=0.04)


def test_sfh_galpop_agrees_with_v0p1_unbounded_params():
    t_table = np.loadtxt(os.path.join(TESTING_DATA_DRN, "t_table_testing_v0.1.0.txt"))
    lgt0 = np.log10(t_table[-1])
    mah_params = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "mah_params_testing_v0.1.0.txt")
    )
    u_ms_params = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_ms_params_testing_v0.1.0.txt")
    )
    u_q_params = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_q_params_testing_v0.1.0.txt")
    )
    sfh_table_v0p1 = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "sfh_table_testing_v0.1.0.txt")
    )
    sfh_table = sfh_galpop(
        t_table,
        mah_params,
        u_ms_params,
        u_q_params,
        lgt0=lgt0,
        fb=FB,
        ms_param_type="unbounded",
        q_param_type="unbounded",
    )
    assert np.allclose(sfh_table_v0p1, sfh_table, rtol=0.05)


def test_ms_param_bounding_agrees_with_v0p1():
    ms_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "ms_params_testing_v0.1.0.txt")
    )
    u_ms_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_ms_params_testing_v0.1.0.txt")
    )
    ms_params_inferred = _get_bounded_sfr_params_vmap(u_ms_params_testing)

    assert np.all(np.isfinite(ms_params_inferred))
    assert np.allclose(ms_params_testing, ms_params_inferred, atol=0.01)


def test_q_param_bounding_agrees_with_v0p1():
    q_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "q_params_testing_v0.1.0.txt")
    )
    u_q_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_q_params_testing_v0.1.0.txt")
    )

    q_params_inferred = _get_bounded_q_params_vmap(u_q_params_testing)
    assert np.all(np.isfinite(q_params_inferred))
    assert np.allclose(q_params_testing, q_params_inferred, atol=0.01)


def test_ms_param_unbounding_agrees_with_v0p1():
    ms_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "ms_params_testing_v0.1.0.txt")
    )
    u_ms_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_ms_params_testing_v0.1.0.txt")
    )
    u_ms_params_inferred = _get_unbounded_sfr_params_vmap(ms_params_testing)
    assert np.all(np.isfinite(u_ms_params_inferred))
    assert np.allclose(u_ms_params_testing, u_ms_params_inferred, rtol=0.01)


def test_q_param_unbounding_agrees_with_v0p1():
    q_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "q_params_testing_v0.1.0.txt")
    )
    u_q_params_testing = np.loadtxt(
        os.path.join(TESTING_DATA_DRN, "u_q_params_testing_v0.1.0.txt")
    )
    u_q_params_inferred = _get_unbounded_q_params_vmap(q_params_testing)
    assert np.all(np.isfinite(u_q_params_inferred))
    assert np.allclose(u_q_params_testing, u_q_params_inferred, rtol=0.01)
