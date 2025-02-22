"""Same as 0D but with varying lambda"""

from pathlib import Path


import utils

here = Path(__file__).absolute().parent
odefile = here / ".." / "odefiles" / "ToRORd_dynCl_endo_zetasplit.ode"
track_names = ("v", "CaTrpn", "TmB", "XU", "J_TRPN", "lmbda", "dLambda", "Zetas", "XS", "cai", "Ta")
Ns = [1, 2, 10, 50]


# Mech step performed every Nth ep step
# Do a set of simulations with various N:
# Ns = np.array([1, 2, 4, 6, 8, 10, 20, 50, 100, 200])


outdir = here / "output-split-zeta"

utils.run_0D(
    odefile=odefile,
    outdir=outdir,
    track_names=track_names,
    Ns=Ns,
    save_traces=True,
    run_full_model=True,
    dt=0.05,
    simdur=500.0,
)
