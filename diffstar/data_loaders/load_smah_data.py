""" """

import os
import warnings

import h5py
import numpy as np

from ..defaults import SFR_MIN
from ..utils import cumulative_mstar_formed_galpop

try:
    from umachine_pyio.load_mock import load_mock_from_binaries

    HAS_UM_LOADER = True
except ImportError:
    HAS_UM_LOADER = False

TASSO = "/Users/aphearin/work/DATA/diffmah_data"
BEBOP = "/lcrc/project/halotools/diffmah_data"
LAPTOP = "/Users/alarcon/Documents/diffmah_data"
BEBOP_SMDPL = os.path.join(
    "/lcrc/project/galsampler/SMDPL/",
    "dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/",
)
BEBOP_SMDPL_DR1 = os.path.join(
    "/lcrc/project/halotools/UniverseMachine/SMDPL/",
    "sfh_binaries_dr1_bestfit/a_1.000000/",
)
H_BPL = 0.678
H_TNG = 0.6774
H_MDPL = H_BPL

# from https://www.cosmosim.org/metadata/smdpl/
FB_SMDPL = 0.048206 / 0.307115  # 0,15696
H_SMDPL = 0.6777

# from https://www.tng-project.org/data/downloads/TNG300-1/
FB_TNG = 0.0486 / 0.3089  # 0,15733
H_TNG = 0.6774


def load_fit_mah_tpeak(basename, data_drn=BEBOP):
    """Load the best fit diffmah parameter data

    Parameters
    ----------
    basename : string
        Name of the h5 file where the diffmah best fit parameters are stored

    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored

    Returns
    -------
    mah_fit_params:  ndarray of shape (n_gal, 4)
        Best fit parameters for each halo:
            (logtc, k, early_index, late_index)

    logm0:  ndarray of shape (n_gal, )
        Diffmah parameter logm0

    """

    fn = os.path.join(data_drn, basename)
    with h5py.File(fn, "r") as hdf:
        mah_fit_params = np.array(
            [
                hdf["logm0"][:],
                hdf["logtc"][:],
                hdf["early_index"][:],
                hdf["late_index"][:],
                hdf["t_peak"][:],
            ]
        ).T
        logm0 = hdf["logm0"][:]

    return mah_fit_params, logm0


def load_fit_sfh(basename, data_drn=BEBOP):
    """Load the best fit diffmah parameter data

    Parameters
    ----------
    basename : string
        Name of the h5 file where the diffmah best fit parameters are stored

    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored

    Returns
    -------
    sfh_fit_params:  ndarray of shape (n_gal, 4)
        Best fit parameters for each halo:
            (logtc, k, early_index, late_index)

    """

    fn = os.path.join(data_drn, basename)
    with h5py.File(fn, "r") as hdf:
        ms_fit_params = np.array(
            [
                hdf["lgmcrit"][:],
                hdf["lgy_at_mcrit"][:],
                hdf["indx_lo"][:],
                hdf["indx_hi"][:],
                hdf["tau_dep"][:],
            ]
        ).T
        q_fit_params = np.array(
            [
                hdf["lg_qt"][:],
                hdf["qlglgdt"][:],
                hdf["lg_drop"][:],
                hdf["lg_rejuv"][:],
            ]
        ).T

    return ms_fit_params, q_fit_params


def load_bolshoi_data(gal_type, data_drn=BEBOP):
    """Load the stellar mass histories from UniverseMachine simulation
    applied to the Bolshoi-Planck (BPL) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_BPL
    from the cosmology of the underlying simulation.

    H_BPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies

    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file

    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation

    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation

    bpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "bpl_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    bpl_t = np.load(os.path.join(data_drn, "bpl_cosmic_time.npy"))

    halo_ids = halos["halo_id"]

    sfrh = halos["sfr_history_main_prog"]
    mstarh = cumulative_mstar_formed_galpop(bpl_t, sfrh)
    log_smahs = np.log10(mstarh)

    return halo_ids, log_smahs, sfrh, bpl_t


def load_bolshoi_small_data(gal_type, data_drn=BEBOP):
    """Load a smaller subsample of the stellar mass histories from the
    UniverseMachine simulation applied to the Bolshoi-Planck (BPL) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_BPL
    from the cosmology of the underlying simulation.

    H_BPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies

    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file

    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation

    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation

    bpl_t : ndarray of shape (n_times, )

        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "um_histories_subsample_dr1_bpl_{}_diffmah.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    bpl_t = np.load(os.path.join(data_drn, "bpl_cosmic_time.npy"))

    halo_ids = halos["halo_id"]

    sfrh = halos["sfr_history_main_prog"]
    mstarh = cumulative_mstar_formed_galpop(bpl_t, sfrh)
    log_smahs = np.log10(mstarh)

    return halo_ids, log_smahs, sfrh, bpl_t


def load_tng_data(data_drn=BEBOP):
    """Load the stellar mass histories from the IllustrisTNG simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_TNG
    from the cosmology of the underlying simulation.

    H_TNG is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies

    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file

    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation

    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation

    tng_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "tng_diffmah.npy"
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))

    halo_ids = np.arange(len(halos["mpeak"])).astype("i8")

    sfrh = halos["sfh"]
    mstarh = cumulative_mstar_formed_galpop(tng_t, sfrh)
    log_smahs = np.log10(mstarh)

    log_mahs = halos["mpeakh"]
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)
    logmp0 = log_mahs[:, -1]

    return halo_ids, log_smahs, sfrh, tng_t, log_mahs, logmp0


def load_tng_small_data(gal_type, data_drn=BEBOP):
    """Load a smaller subsample of the stellar mass histories from
    the IllustrisTNG simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_TNG
    from the cosmology of the underlying simulation.

    H_TNG is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation.
    tng_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "tng_small.npy"
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))

    halo_ids = np.arange(len(halos["mpeak"])).astype("i8")

    sfrh = halos["sfh"]
    mstarh = cumulative_mstar_formed_galpop(tng_t, sfrh)
    log_smahs = np.log10(mstarh)

    return halo_ids, log_smahs, sfrh, tng_t


def load_mdpl_data(gal_type, data_drn=BEBOP):
    """Load the stellar mass histories from the UniverseMachine simulation
    applied to the MultiDark Planck 2 (MDPL2) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_MDPL
    from the cosmology of the underlying simulation.

    H_MDPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation.
    mdpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "mdpl2_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    mdpl_t = np.load(os.path.join(data_drn, "mdpl2_cosmic_time.npy"))

    halo_ids = halos["halo_id"]

    sfrh = halos["sfr_history_main_prog"]
    mstarh = cumulative_mstar_formed_galpop(mdpl_t, sfrh)
    log_smahs = np.log10(mstarh)

    return halo_ids, log_smahs, sfrh, mdpl_t


def load_mdpl_small_data(gal_type, data_drn=BEBOP):
    """Load a smaller subsample of the stellar mass histories from the
    UniverseMachine simulation applied to the
    MultiDark Planck 2 (MDPL2) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_MDPL
    from the cosmology of the underlying simulation.

    H_MDPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation.
    mdpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    basename = "um_histories_dr1_mdpl2_small_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    mdpl_t = np.load(os.path.join(data_drn, "mdpl2_cosmic_time.npy"))

    halo_ids = halos["halo_id"]

    sfrh = halos["sfr_history_main_prog"]
    mstarh = cumulative_mstar_formed_galpop(mdpl_t, sfrh)
    log_smahs = np.log10(mstarh)

    return halo_ids, log_smahs, sfrh, mdpl_t


def load_SMDPL_nomerging_data(subvols, data_drn=BEBOP_SMDPL):
    """Load the stellar mass histories from UniverseMachine simulation
    applied to the Bolshoi-Planck (BPL) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_BPL
    from the cosmology of the underlying simulation.

    H_BPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h in the simulation.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h in the simulation.
    bpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    if not HAS_UM_LOADER:
        raise ImportError("Must have umachine_pyio installed to load this dataset")

    galprops = ["halo_id", "sfr_history_main_prog", "mpeak_history_main_prog"]
    mock = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

    SMDPL_t = np.loadtxt(os.path.join(data_drn, "smdpl_cosmic_time.txt"))

    halo_ids = mock["halo_id"]

    sfrh = mock["sfr_history_main_prog"]
    mstarh = cumulative_mstar_formed_galpop(SMDPL_t, sfrh)
    log_smahs = np.log10(mstarh)

    lgmh_min = 7.0
    mh_min = 10**lgmh_min
    msk = mock["mpeak_history_main_prog"] < mh_min
    clipped_mahs = np.where(msk, 1.0, mock["mpeak_history_main_prog"])
    log_mahs = np.log10(clipped_mahs)
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)

    logmp0 = log_mahs[:, -1]

    return halo_ids, log_smahs, sfrh, SMDPL_t, log_mahs, logmp0


def load_SMDPL_DR1_data(subvols, data_drn=BEBOP_SMDPL_DR1):
    """Load the stellar mass histories from UniverseMachine simulation
    applied to the Bolshoi-Planck (BPL) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_BPL
    from the cosmology of the underlying simulation.

    H_BPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h=0.67.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h=0.67.
    bpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    """
    if not HAS_UM_LOADER:
        raise ImportError("Must have umachine_pyio installed to load this dataset")

    galprops = ["halo_id", "sfr_history_all_prog", "mpeak_history_main_prog"]
    mock = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

    SMDPL_t = np.loadtxt(os.path.join(data_drn, "smdpl_cosmic_time.txt"))

    halo_ids = mock["halo_id"]

    sfrh = mock["sfr_history_all_prog"]
    mstarh = cumulative_mstar_formed_galpop(SMDPL_t, sfrh)
    log_smahs = np.log10(mstarh)

    lgmh_min = 7.0
    mh_min = 10**lgmh_min
    msk = mock["mpeak_history_main_prog"] < mh_min
    clipped_mahs = np.where(msk, 1.0, mock["mpeak_history_main_prog"])
    log_mahs = np.log10(clipped_mahs)
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)

    logmp0 = log_mahs[:, -1]

    return halo_ids, log_smahs, sfrh, SMDPL_t, log_mahs, logmp0
