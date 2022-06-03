"""
"""
import numpy as np
import os
import warnings
from umachine_pyio.load_mock import load_mock_from_binaries
from astropy.cosmology import Planck15


BEBOP = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_z0_binaries/"

H_BPL = 0.678

def load_fit_mah(filename, data_drn=BEBOP):
    """ Load the best fit diffmah parameter data.
    Parameters
    ----------
    filename : string
        Name of the h5 file where the diffmah best fit parameters are stored.
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    mah_fit_params:  ndarray of shape (n_gal, 4)
        Best fit parameters for each halo:
            (logtc, k, early_index, late_index)
    logmp:  ndarray of shape (n_gal, )
        Base-10 logarithm of the present day peak halo mass.
    """
    fitting_data = dict()

    fn = os.path.join(data_drn, filename)
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            if key == "halo_id":
                fitting_data[key] = hdf[key][...]
            else:
                fitting_data["fit_" + key] = hdf[key][...]


def load_SMDPL_data(subvols, data_drn=BEBOP):
    galprops = ["halo_id", "mpeak_history_main_prog"]
    _halos = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)
    halo_ids = np.array(_halos["halo_id"])
    _mah = np.maximum.accumulate(_halos["mpeak_history_main_prog"], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

    # Needed for .../SMDPL/sfr_catalogs 
    # but not for .../SMDPL/sfh_z0_binaries 
    # log_mahs = np.where(log_mahs > 0.0, log_mahs + np.log10(H_BPL), log_mahs)

    
    SMDPL_a = np.load('/lcrc/project/halotools/UniverseMachine/SMDPL/scale_list.npy')
    SMDPL_z = 1.0 / SMDPL_a - 1.0
    SMDPL_t = Planck15.age(SMDPL_z).value

    log_mah_fit_min = 10.0
    return halo_ids, log_mahs, SMDPL_t, log_mah_fit_min

