"""
"""
from collections import OrderedDict

DEFAULT_MS_PARAMS = OrderedDict(
    lgmcrit=12.0,
    lgy_at_mcrit=-1.0,
    indx_lo=1.0,
    indx_hi=-1.0,
    tau_dep=2.0,
)


DEFAULT_Q_PARAMS = OrderedDict(
    u_lg_qt=1.0, u_lg_qs=-0.3, u_lg_drop=-1.0, u_lg_rejuv=-0.5
)
