"""
"""
import numpy as np
import os
import h5py
import warnings
from diffstar.utils import _get_dt_array

TASSO = "/Users/aphearin/work/DATA/diffmah_data"
BEBOP = "/lcrc/project/halotools/diffmah_data"
LAPTOP = "/Users/alarcon/Documents/diffmah_data"


def load_fit_mah(filename, data_drn=BEBOP):
    """ Load the best fit diffmah parameter data.
    """
    fitting_data = dict()

    fn = os.path.join(data_drn, filename)
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            if key == "halo_id":
                fitting_data[key] = hdf[key][...]
            else:
                fitting_data["fit_" + key] = hdf[key][...]

    mah_fit_params = np.array(
        [
            fitting_data["fit_mah_logtc"],
            fitting_data["fit_mah_k"],
            fitting_data["fit_early_index"],
            fitting_data["fit_late_index"],
        ]
    ).T
    logmp = fitting_data["fit_logmp_fit"]

    return mah_fit_params, logmp


def load_bolshoi_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "bpl_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    bpl_t = np.load(os.path.join(data_drn, "bpl_cosmic_time.npy"))

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(bpl_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, bpl_t, dt


def load_bolshoi_small_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "um_histories_subsample_dr1_bpl_{}_diffmah.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    bpl_t = np.load(os.path.join(data_drn, "bpl_cosmic_time.npy"))

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(bpl_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, bpl_t, dt


def load_tng_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "tng_diffmah.npy"
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))

    halo_ids = np.arange(len(halos["mpeak"])).astype("i8")
    dt = _get_dt_array(tng_t)
    sfrh = halos["sfh"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, tng_t, dt


def load_tng_small_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "tng_small.npy"
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))

    halo_ids = np.arange(len(halos["mpeak"])).astype("i8")
    dt = _get_dt_array(tng_t)
    sfrh = halos["sfh"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, tng_t, dt


def load_mdpl_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "mdpl2_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    mdpl_t = np.load(os.path.join(data_drn, "mdpl2_cosmic_time.npy"))

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(mdpl_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, mdpl_t, dt


def load_mdpl_small_data(gal_type, data_drn=BEBOP):
    """ Load the stellar mass histories from BPL
    """
    basename = "um_histories_dr1_mdpl2_small_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    halos = np.load(fn)
    mdpl_t = np.load(os.path.join(data_drn, "mdpl2_cosmic_time.npy"))

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(mdpl_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, mdpl_t, dt
