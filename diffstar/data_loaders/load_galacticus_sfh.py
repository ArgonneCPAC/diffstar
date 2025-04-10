""" """

from . import load_flat_hdf5


def load_galacticus_sfh_data_block(fn, istart, iend):
    tarr_dict = load_flat_hdf5(fn, keys=["tarr"])
    block_keys = (
        "sfh_in_situ_bulge",
        "sfh_in_situ_disk",
        "sfh_tot_bulge",
        "sfh_tot_disk",
    )
    sfh_data_block = load_flat_hdf5(fn, istart=istart, iend=iend, keys=block_keys)
    sfh_data_block["tarr"] = tarr_dict["tarr"]

    return sfh_data_block
