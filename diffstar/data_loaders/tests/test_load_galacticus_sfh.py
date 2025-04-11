""" """

import os

import pytest

from .. import load_galacticus_sfh as lgs

HAS_TESTING_DATA = False
if os.path.isdir(lgs.DRN_POBOY):
    HAS_TESTING_DATA = True

MSG_HAS_NO_DATA = "Must have diffstar fit data on local disk to run this test"


@pytest.mark.skipif(not HAS_TESTING_DATA, reason=MSG_HAS_NO_DATA)
def test_load_galacticus_diffstar_data():
    diffstar_fit_data = lgs.load_galacticus_diffstar_data(lgs.DRN_POBOY)
    diffmah_loss = diffstar_fit_data.diffmah_fit_data["loss"]
    n_halos = diffmah_loss.size
    diffstar_loss_in_situ = diffstar_fit_data.diffstar_in_situ_fit_data["loss"]
    n_gals = diffstar_loss_in_situ.size
    diffstar_loss_tot_sfh = diffstar_fit_data.diffstar_in_plus_ex_situ_fit_data["loss"]
    n_gals2 = diffstar_loss_tot_sfh.size
    assert n_halos == n_gals == n_gals2
