"""
"""
import typing
import numpy as np
from astropy.io import fits

BNAME = "des_y3_rsmodel_v0.8.1.fits"


class RedSequenceData(typing.NamedTuple):
    """NamedTuple with info about RedMapper Red Sequence for DES Y3"""

    redshift: np.ndarray
    gr_0: np.ndarray
    ri_0: np.ndarray
    iz_0: np.ndarray
    gr_1: np.ndarray
    ri_1: np.ndarray
    iz_1: np.ndarray
    mz_0: np.ndarray
    covmat: np.ndarray


def load_des_y3_data(fn, z_lo=0.105, z_hi=0.8):
    """<c | m_z, redshift> = c_0(redshift) + c_1(redshift) * (z - z_0(redshift))"""
    with fits.open(fn) as hdulist:
        arr = np.array(hdulist[1].data)

    istart = np.searchsorted(arr["z"], z_lo)
    iend = np.searchsorted(arr["z"], z_hi)

    redshift = arr["z"][istart:iend]

    gr_0 = arr["c"][istart:iend, 0]
    ri_0 = arr["c"][istart:iend, 1]
    iz_0 = arr["c"][istart:iend, 2]

    gr_1 = arr["slope"][istart:iend, 0]
    ri_1 = arr["slope"][istart:iend, 1]
    iz_1 = arr["slope"][istart:iend, 2]

    mz_0 = arr["pivotmag"][istart:iend]

    covmat = arr["covmat"][istart:iend, :, :]
    # Ignore off-diagonal entries which Eli says are arbitrary
    covmat = covmat * np.eye(3)

    return RedSequenceData(redshift, gr_0, ri_0, iz_0, gr_1, ri_1, iz_1, mz_0, covmat)


def compute_rs_locus(
    redshift_obs, m_z, redshift_table, c_0_table, c_1_table, z_0_table
):
    """Calculate red sequence locus by interpolating from RedSequenceData"""
    c_0 = np.interp(redshift_obs, redshift_table, c_0_table)
    c_1 = np.interp(redshift_obs, redshift_table, c_1_table)
    m_z_0 = np.interp(redshift_obs, redshift_table, z_0_table)
    return c_0 + c_1 * (m_z - m_z_0)
