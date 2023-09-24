"""
"""


def test_get_sfh_from_mah_kern_imports_from_top_level_kernels():
    from ...kernels import get_sfh_from_mah_kern

    get_sfh_from_mah_kern()
