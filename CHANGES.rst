0.3.2 (2024-10-25)
------------------
- Adapt sfh kernels to be compatible with diffmah 0.6.1


0.3.1 (2024-6-19)
------------------
- Performance improvements for calc_sfh_galpop and calc_sfh_singlegal


0.3.0 (2024-01-17)
------------------
- Implement new API for primary user-facing functions calc_sfh_galpop and calc_sfh_singlegal


0.2.4 (2024-01-16)
------------------
- Require diffmah>=0.5.0


0.2.3 (2024-01-15)
------------------
- Switch to namedtuple for diffstar parameters (https://github.com/ArgonneCPAC/diffstar/pull/45)


0.2.2 (2023-10-08)
------------------
- Fix forgotten import so that diffstar.__version__ now works


0.2.1 (2023-09-27)
------------------
- Implement floor at SFR_MIN in all computations of SFH


0.2.0 (2023-09-25)
------------------
- Remove quenching.py module
- Add new kernel_builders.py module
- sfh.py now contains user-facing functions sfh_singlegal and sfh_galpop


0.1.0 (2022-08-29)
------------------
- First release