"""
save_loss_data.py

Usage (inside Python):
    from save_loss_data import save_loss_data_h5
    save_loss_data_h5("unit_test_loss_data.h5",
                      loss_data_mstar,
                      loss_data_ssfr,
                      loss_data_ssfr_sat)

This expects you already have the three tuples in memory, in the exact orders shown
in your message.
"""

from typing import Any, Iterable, Sequence
import numpy as np
import h5py
import json

from fit_get_loss_helpers_mgash import (
    get_loss_data_smhm,
    get_loss_data_pdfs_mstar,
    get_loss_data_pdfs_ssfr_central,
    get_loss_data_pdfs_ssfr_satellite,
)

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
    "lgt0",
    "fb",
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
    "lgt0",
    "fb",
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
    "lgt0",
    "fb",
    "ndbins_lo",
    "ndbins_hi",
    "logmstar_bins_pdf",
    "logssfr_bins_pdf",
    "mhalo_pdf_sat_ragged",  # ragged allowed
    "indx_pdf",
    "target_mstar_ids",
    "target_data_sat",
]


def _to_numpy(x: Any):
    """Convert JAX/array-like -> NumPy array without copying if possible."""
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return x  # leave as-is (e.g., a list of arrays)


def _is_array_like(x: Any) -> bool:
    return isinstance(x, (np.ndarray,)) or hasattr(x, "__array__")


def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _is_ragged_sequence(seq: Sequence[Any]) -> bool:
    """
    Heuristic: sequence of array-like where shapes are not all equal.
    Also treat object-dtype arrays as ragged.
    """
    if isinstance(seq, np.ndarray) and seq.dtype == object:
        return True
    if not _is_sequence(seq):
        return False
    shapes = []
    for el in seq:
        if _is_array_like(el):
            a = _to_numpy(el)
            shapes.append(a.shape)
        else:
            # Non-array element -> treat as ragged to be safe
            return True
    return len(set(shapes)) > 1


def _save_value(group: h5py.Group, name: str, value: Any):
    """
    Save a single item under group/name.
    - Regular ndarrays: one dataset
    - Scalar: 0-D dataset
    - Ragged sequences: a subgroup with datasets '0000', '0001', ...
    - Sequence with equal shapes: stacked into one dataset
    """
    # If object-dtype NumPy => likely ragged
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = list(value)  # treat as ragged sequence

    if _is_sequence(value) and _is_ragged_sequence(value):
        # Save as subgroup with one dataset per element
        sub = group.create_group(name)
        for i, el in enumerate(value):
            arr = _to_numpy(el)
            sub.create_dataset(f"{i:04d}", data=arr)
        sub.attrs["format"] = "ragged_list_of_datasets"
        sub.attrs["length"] = len(value)
        return

    # Non-ragged sequences of equal-shaped arrays -> stack
    if _is_sequence(value):
        # Convert to array if possible (will stack)
        try:
            arr = _to_numpy(value)
            group.create_dataset(name, data=arr)
            return
        except Exception:
            # Fallback: save as subgroup
            sub = group.create_group(name)
            for i, el in enumerate(value):
                arr = _to_numpy(el)
                sub.create_dataset(f"{i:04d}", data=arr)
            sub.attrs["format"] = "list_of_datasets"
            sub.attrs["length"] = len(value)
            return

    # Scalar or array-like
    if _is_array_like(value):
        arr = _to_numpy(value)
        group.create_dataset(name, data=arr)
        return

    # Last resort: store JSON-serializable objects as attrs
    try:
        group.attrs[name] = json.dumps(value)
    except Exception:
        # If we end up here, user passed a very custom object
        # Save a string repr so the test can still load something.
        group.attrs[name] = repr(value)


def save_loss_data_h5(
    filename: str,
    loss_data_mstar: Iterable[Any],
    loss_data_ssfr: Iterable[Any],
    loss_data_ssfr_sat: Iterable[Any],
):
    """
    Save three loss-data tuples into an HDF5 file with a clean hierarchy:

        /loss_data_mstar/...
        /loss_data_ssfr/...
        /loss_data_ssfr_sat/...

    Each dataset is named after the variable (e.g., 'mah_params_data').
    Ragged arrays/lists are stored as a subgroup with one dataset per element.

    Parameters
    ----------
    filename : str
        Output .h5 path.
    loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat : tuple-like
        Tuples exactly matching the field orders defined above.
    """
    # Sanity checks on tuple lengths
    if len(loss_data_mstar) != len(MSTAR_FIELDS):
        raise ValueError(
            f"loss_data_mstar length {len(loss_data_mstar)} != {len(MSTAR_FIELDS)}"
        )
    if len(loss_data_ssfr) != len(SSFR_FIELDS):
        raise ValueError(
            f"loss_data_ssfr length {len(loss_data_ssfr)} != {len(SSFR_FIELDS)}"
        )
    if len(loss_data_ssfr_sat) != len(SSFR_SAT_FIELDS):
        raise ValueError(
            f"loss_data_ssfr_sat length {len(loss_data_ssfr_sat)} != {len(SSFR_SAT_FIELDS)}"
        )

    with h5py.File(filename, "w") as f:
        # mstar
        g_m = f.create_group("loss_data_mstar")
        g_m.attrs["field_order"] = json.dumps(MSTAR_FIELDS)
        for name, val in zip(MSTAR_FIELDS, loss_data_mstar):
            _save_value(g_m, name, val)

        # ssfr (centrals)
        g_c = f.create_group("loss_data_ssfr")
        g_c.attrs["field_order"] = json.dumps(SSFR_FIELDS)
        for name, val in zip(SSFR_FIELDS, loss_data_ssfr):
            _save_value(g_c, name, val)

        # ssfr (satellites)
        g_s = f.create_group("loss_data_ssfr_sat")
        g_s.attrs["field_order"] = json.dumps(SSFR_SAT_FIELDS)
        for name, val in zip(SSFR_SAT_FIELDS, loss_data_ssfr_sat):
            _save_value(g_s, name, val)

        # File-level note to help future you
        f.attrs["description"] = (
            "Unit testing data for mstar/ssfr kernels. Ragged lists are stored as groups "
            "with one dataset per element and 'format' attr."
        )

    print(f"Wrote {filename}")


# --- Optional: small loader helper for ragged groups (use in tests) ---


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


indir = (
    "/Users/alarcon/Documents/diffmah_data/mgash/smdpl_dr1_nomerging_pdf_target_data/"
)
nhalos = 10
loss_data_mstar, plot_data_pdf = get_loss_data_pdfs_mstar(indir, nhalos)
loss_data_ssfr, plot_data_pdf_ssfr_cen = get_loss_data_pdfs_ssfr_central(indir, nhalos)
loss_data_ssfr_sat, plot_data_pdf_ssfr_sat = get_loss_data_pdfs_ssfr_satellite(
    indir, nhalos
)
fname = "loss_kernels_testing_data_10halos.h5"
save_loss_data_h5(fname, loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat)


nhalos = 100
loss_data_mstar, plot_data_pdf = get_loss_data_pdfs_mstar(indir, nhalos)
loss_data_ssfr, plot_data_pdf_ssfr_cen = get_loss_data_pdfs_ssfr_central(indir, nhalos)
loss_data_ssfr_sat, plot_data_pdf_ssfr_sat = get_loss_data_pdfs_ssfr_satellite(
    indir, nhalos
)
fname = "loss_kernels_testing_data_100halos.h5"
save_loss_data_h5(fname, loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat)
