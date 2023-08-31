"""
"""
from ..fit_smah_helpers import get_header


def test_get_header():
    header = get_header()
    assert "halo_id" in header
