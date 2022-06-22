from umachine_pyio.load_mock import load_mock_from_binaries
import numpy as np
import warnings

galprops = ["halo_id", "sfr_history_main_prog", "mpeak_history_main_prog"]
BEBOP_SMDPL = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_z0_binaries/"
data_drn = BEBOP_SMDPL
H_SMDPL = 0.6777

# subvols = np.array([1])
subvols = np.arange(10)
halos = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

halo_ids = halos["halo_id"]
sfrh = halos["sfr_history_main_prog"]
_mah = np.maximum.accumulate(halos["mpeak_history_main_prog"], axis=1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

particle_mass_res = 9.63e7 / H_SMDPL
# So we cut halos with M0 below 500 times the mass resolution.
logmpeak_fit_min = np.log10(500 * particle_mass_res)
logmpeak = log_mahs[:, -1]

sel = logmpeak >= logmpeak_fit_min

_mah = _mah[sel]
sfrh = sfrh[sel]
halo_ids = halo_ids[sel]


np.savez(
    "/lcrc/project/halotools/alarcon/data/SMDPL_subvol0..9.npz",
    halo_id=halo_ids,
    sfr_history_main_prog=sfrh,
    mpeak_history_main_prog=_mah,
)
