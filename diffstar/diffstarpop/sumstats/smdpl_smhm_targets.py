"""
"""

import os

import h5py

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def umachine_smhm_z0_allhalos():
    drn = os.path.join(_THIS_DRNAME, "tests", "testing_data")
    bn = "smdpl_smhm_z0.h5"
    fn = os.path.join(drn, bn)
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]

    return data["logmh_bins"], data["smhm"]
