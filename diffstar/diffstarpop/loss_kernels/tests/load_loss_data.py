import h5py
import json

# --- Names in tuple order (so we can label datasets clearly) ---

MSTAR_FIELDS = [
    "mah_params_data",
    "logmp0_data",
    "upid_data",
    "lgmu_infall_data",
    "logmhost_infall_data",
    "gyr_since_infall_data",
    "ran_key_data",
    "t_obs_targets",
    "logmstar_bins_pdf",
    "mstar_counts_target",
]

SSFR_FIELDS = [
    "mah_params_data",
    "logmp0_data",
    "upid_data",
    "lgmu_infall_data",
    "logmhost_infall_data",
    "gyr_since_infall_data",
    "ran_key_data",
    "t_obs_targets",
    "ndbins_lo",
    "ndbins_hi",
    "logmstar_bins_pdf",
    "logssfr_bins_pdf",
    "mhalo_pdf_cen_ragged",  # ragged allowed
    "indx_pdf",
    "target_mstar_ids",
    "target_data",
]

SSFR_SAT_FIELDS = [
    "mah_params_data",
    "logmp0_data",
    "upid_data",
    "lgmu_infall_data",
    "logmhost_infall_data",
    "gyr_since_infall_data",
    "ran_key_data",
    "t_obs_targets",
    "ndbins_lo",
    "ndbins_hi",
    "logmstar_bins_pdf",
    "logssfr_bins_pdf",
    "mhalo_pdf_sat_ragged",  # ragged allowed
    "indx_pdf",
    "target_mstar_ids",
    "target_data_sat",
]


def load_loss_data_h5(filename: str):
    """
    Load the three tuples back from disk. Ragged groups are returned as lists
    of NumPy arrays. Returns (loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat).
    """

    def _load_group(g: h5py.Group, field_names):
        out = []
        for name in field_names:
            if name in g:
                obj = g[name]
                if isinstance(obj, h5py.Dataset):
                    out.append(obj[()])
                elif isinstance(obj, h5py.Group):
                    fmt = (
                        obj.attrs.get("format", "").decode()
                        if isinstance(obj.attrs.get("format", ""), bytes)
                        else obj.attrs.get("format", "")
                    )
                    if fmt in ("ragged_list_of_datasets", "list_of_datasets"):
                        items = [obj[k][()] for k in sorted(obj.keys())]
                        out.append(items)
                    else:
                        # Unknown layout—try datasets in key order
                        items = [obj[k][()] for k in sorted(obj.keys())]
                        out.append(items)
                else:
                    raise RuntimeError(f"Unexpected HDF5 object at {g.name}/{name}")
            else:
                # Might be stored as attr (rare)
                if name in g.attrs:
                    val = g.attrs[name]
                    try:
                        out.append(json.loads(val))
                    except Exception:
                        out.append(val)
                else:
                    raise KeyError(f"Field '{name}' not found in group {g.name}")
        return tuple(out)

    with h5py.File(filename, "r") as f:
        m_fields = json.loads(f["loss_data_mstar"].attrs["field_order"])
        c_fields = json.loads(f["loss_data_ssfr"].attrs["field_order"])
        s_fields = json.loads(f["loss_data_ssfr_sat"].attrs["field_order"])

        mstar = _load_group(f["loss_data_mstar"], m_fields)
        ssfr = _load_group(f["loss_data_ssfr"], c_fields)
        ssfr_s = _load_group(f["loss_data_ssfr_sat"], s_fields)

    return mstar, ssfr, ssfr_s
