from pathlib import Path
import gotranx
import numpy as np
import matplotlib.pyplot as plt


def twitch(t, tstart=0.05, ca_ampl=-0.2):
    tau1 = 0.05 * 1000
    tau2 = 0.110 * 1000

    ca_diast = 0.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1) - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca + 1.0


def update_lambda_and_dlambda(t, prev_lmbda, dt):
    lmbda_ti = twitch(t)
    p[lmbda_index] = lmbda_ti
    p_mechanics[lmbda_index_mechanics] = lmbda_ti
    p_ep[lmbda_index_ep] = lmbda_ti

    dLambda = (lmbda_ti - prev_lmbda) / dt
    p[dLambda_index] = dLambda
    p_mechanics[dLambda_index_mechanics] = dLambda
    p_ep[dLambda_index_ep] = dLambda
    prev_lmbda = lmbda_ti
    return p, p_mechanics, p_ep, prev_lmbda


# Load the model
ode = gotranx.load_ode("ORdmm_Land.ode")

mechanics_comp = ode.get_component("mechanics")
mechanics_ode = mechanics_comp.to_ode()


ep_ode = ode - mechanics_comp
ep_file = Path("ORdmm_Land_ep.py")


# Generate model code from .ode file
rebuild = False
if not ep_file.is_file() or rebuild:
    # Generate code for full model. The full model output is plotted together with the splitting
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
    )

    # Generate code for the electrophysiology model
    code_ep = gotranx.cli.gotran2py.get_code(
        ep_ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
        missing_values=mechanics_ode.missing_variables,
    )

    # Generate code for the mechanics model
    code_mechanics = gotranx.cli.gotran2py.get_code(
        mechanics_ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
        missing_values=ep_ode.missing_variables,
    )

    # Create ep, mechanics and full model to files:
    ep_file.write_text(code_ep)
    Path("ORdmm_Land_mechanics.py").write_text(code_mechanics)
    Path("ORdmm_Land.py").write_text(code)


# Import ep, mechanics and full model
import ORdmm_Land_ep
import ORdmm_Land_mechanics
import ORdmm_Land

model = ORdmm_Land.__dict__
ep_model = ORdmm_Land_ep.__dict__
mechanics_model = ORdmm_Land_mechanics.__dict__

# Set time step to 0.1 ms
dt = 0.1  # 05
simdur = 700  # Simulation duration
t = np.arange(0, simdur, dt)


# Get the index of the membrane potential
V_index_ep = ep_model["state_index"]("v")
# Forwared generalized rush larsen scheme for the electrophysiology model
fgr_ep = ep_model["forward_generalized_rush_larsen"]
# Monitor function for the electrophysiology model
mon_ep = ep_model["monitor_values"]
# Missing values function for the electrophysiology model
mv_ep = ep_model["missing_values"]
# Index of the calcium concentration
Ca_index_ep = ep_model["state_index"]("cai")


CaTrpn_index_ep = ep_model["state_index"]("CaTrpn")
dLambda_index_ep = ep_model["parameter_index"]("dLambda")
lmbda_index_ep = ep_model["parameter_index"]("lmbda")

# From split-cai 0D (not in zeta 3D):
# Forwared generalized rush larsen scheme for the mechanics model
fgr_mechanics = mechanics_model["forward_generalized_rush_larsen"]
# Monitor function for the mechanics model
mon_mechanics = mechanics_model["monitor_values"]
# Missing values function for the mechanics model
mv_mechanics = mechanics_model["missing_values"]

Ta_index_mechanics = mechanics_model["monitor_index"]("Ta")
J_TRPN_index_ep = ep_model["monitor_index"]("J_TRPN")
XS_index_ep = ep_model["state_index"]("XS")
TmB_index_ep = ep_model["state_index"]("TmB")
XU_index_ep = ep_model["monitor_index"]("XU")


lmbda_index_mechanics = mechanics_model["parameter_index"]("lmbda")
Zetas_index_mechanics = mechanics_model["state_index"]("Zetas")
dLambda_index_mechanics = mechanics_model["parameter_index"]("dLambda")


# From split-cai 0D (not in zeta 3D):
# Forwared generalized rush larsen scheme for the full model
fgr = model["forward_generalized_rush_larsen"]
# Monitor function for the full model
mon = model["monitor_values"]
Ta_index = model["monitor_index"]("Ta")
J_TRPN_index = model["monitor_index"]("J_TRPN")
CaTrpn_index = model["state_index"]("CaTrpn")
TmB_index = model["state_index"]("TmB")
XU_index = model["monitor_index"]("XU")

lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")
XS_index = model["state_index"]("XS")
Zetas_index = model["state_index"]("Zetas")


# Tolerances to test for when to perform steps in the mechanics model
tols = [
    1e-15,
    5e-4,
    1e-3,
    2.5e-3,
    5e-3,
    7.5e-3,
    1e-2,
    2.5e-2,
    5e-2,
]


# Colors for the plots
from itertools import cycle

colors = cycle(["r", "g", "b", "c", "m"])
linestyles = cycle(["-", "--", "-.", ":"])

# Create arrays to store the results
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))
CaTrpn_ep = np.zeros(len(t))

CaTrpn_full = np.zeros(len(t))
J_TRPN_full = np.zeros(len(t))
Ta_full = np.zeros(len(t))
Zetas_full = np.zeros(len(t))
XS_full = np.zeros(len(t))
dLambda_full = np.zeros(len(t))
TmB_full = np.zeros(len(t))
XU_full = np.zeros(len(t))

Ta_mechanics = np.zeros(len(t))
J_TRPN_ep = np.zeros(len(t))
lmbda_mechanics = np.zeros(len(t))
Zetas_mechanics = np.zeros(len(t))
Zetaw_mechanics = np.zeros(len(t))
dLambda_mechanics = np.zeros(len(t))

XS_ep = np.zeros(len(t))
TmB_ep = np.zeros(len(t))
XU_ep = np.zeros(len(t))


fig, ax = plt.subplots(9, 2, sharex=True, figsize=(14, 10))
lines = []
labels = []


for j, (col, ls, tol) in enumerate(zip(colors, linestyles, tols)):
    # Get initial values from the EP model
    y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"]()
    ep_missing_values = np.repeat(0.0001, len(ep_ode.missing_variables))

    # From split-cai 0D (not in zeta 3D):
    # Get initial values from the mechanics model
    y_mechanics = mechanics_model["init_state_values"]()
    p_mechanics = mechanics_model["init_parameter_values"]()
    mechanics_missing_values = np.repeat(0.0001, len(mechanics_ode.missing_variables))

    # Get the initial values from the full model
    y = model["init_state_values"]()
    p = model["init_parameter_values"]()

    # TODO: figure out what the alternative is for this:
    # The mechanics missing varables are XS and XW, which are states. Do the same as with cai?
    # In the 3D case, they were also initialized to zero
    mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
    ep_missing_values[:] = mv_mechanics(0, y_mechanics, p_mechanics, mechanics_missing_values)

    # We will store the previous missing values to check for convergence
    prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
    prev_mechanics_missing_values[:] = mechanics_missing_values

    inds = []
    count = 1
    max_count = 10
    prev_lmbda = p[lmbda_index]
    p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(np.float64(0), prev_lmbda, dt)
    for i, ti in enumerate(t):
        # Forward step for the full model
        y[:] = fgr(y, ti, dt, p)
        monitor = mon(ti, y, p)
        J_TRPN_full[i] = monitor[J_TRPN_index]
        Ta_full[i] = monitor[Ta_index]
        XS_full[i] = y[XS_index]
        Zetas_full[i] = y[Zetas_index]
        dLambda_full[i] = p[dLambda_index]
        CaTrpn_full[i] = y[CaTrpn_index]
        TmB_full[i] = y[TmB_index]
        XU_full[i] = monitor[XU_index]

        # Forward step for the EP model (from cai split)
        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        CaTrpn_ep[i] = y_ep[CaTrpn_index_ep]
        monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values)
        TmB_ep[i] = y_ep[TmB_index_ep]
        XU_ep[i] = monitor_ep[XU_index_ep]
        J_TRPN_ep[i] = monitor_ep[J_TRPN_index_ep]
        XS_ep[i] = y_ep[XS_index_ep]

        # Update missing values for the mechanics model
        mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values)

        # Compute the change in the missing values
        change = np.linalg.norm(
            mechanics_missing_values - prev_mechanics_missing_values
        ) / np.linalg.norm(prev_mechanics_missing_values)

        # Check if the change is small enough to continue to the next time step
        if change < tol:
            count += 1
            # Very small change to just continue to next time step
            if count < max_count:
                p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(
                    ti + dt, prev_lmbda, dt
                )
                continue

        # Store the index of the time step where we performed a step
        inds.append(i)

        # Forward step for the mechanics model
        # y_mechanics[:] = fgr_mechanics(
        #    y_mechanics, ti, count * dt, p_mechanics, mechanics_missing_values
        # )
        y_mechanics[:] = fgr_mechanics(
            y_mechanics, ti, count * dt, p_mechanics, prev_mechanics_missing_values
        )
        count = 1
        monitor_mechanics = mon_mechanics(
            ti,
            y_mechanics,
            p_mechanics,
            mechanics_missing_values,
        )

        # lambda_min12 = monitor_mechanics[lambda_min12_index]
        # print(lambda_min12)

        Ta_mechanics[i] = monitor_mechanics[Ta_index_mechanics]
        Zetas_mechanics[i] = y_mechanics[Zetas_index_mechanics]
        dLambda_mechanics[i] = p_mechanics[dLambda_index_mechanics]
        lmbda_mechanics[i] = p_mechanics[lmbda_index_mechanics]

        p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(ti + dt, prev_lmbda, dt)
        # Update missing values for the EP model
        ep_missing_values[:] = mv_mechanics(t, y_mechanics, p_mechanics, mechanics_missing_values)

        prev_mechanics_missing_values[:] = mechanics_missing_values

    # Plot the results
    perc = 100 * len(inds) / len(t)
    print(f"Solved on {perc}% of the time steps")
    inds = np.array(inds)

    if j == 0:
        # Plot the full model with a dashed line only for the first run
        (l,) = ax[8, 0].plot(t, Ta_full, color="k", linestyle="--", label="Full")
        ax[2, 0].plot(t, J_TRPN_full, color="k", linestyle="--", label="Full")
        ax[3, 0].plot(t, CaTrpn_full, color="k", linestyle="--", label="Full")
        ax[4, 0].plot(t, TmB_full, color="k", linestyle="--", label="Full")
        ax[5, 0].plot(t, XU_full, color="k", linestyle="--", label="Full")
        ax[6, 0].plot(t, XS_full, color="k", linestyle="--", label="Full")
        ax[7, 0].plot(t, Zetas_full, color="k", linestyle="--", label="Full")

        lines.append(l)
        labels.append("Full")

    (l,) = ax[0, 0].plot(t, V_ep, color=col, linestyle=ls, label=f"tol={tol}")
    lines.append(l)
    labels.append(f"tol={tol}, perc={perc:.2f}%")

    ax[0, 1].plot(
        t,
        Ca_ep,
        color=col,
        linestyle=ls,
        # label=f"tol={tol}",
    )

    ax[0, 0].set_ylabel("V (mV)")
    ax[0, 1].set_ylabel("Ca (mM)")

    ax[1, 0].plot(
        t[inds],
        lmbda_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
        # marker='.'
    )

    ax[1, 1].plot(
        t[inds],
        dLambda_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[1, 0].set_ylabel("Lambda mech")
    ax[1, 1].set_ylabel("dLambda mech")

    err_Ta = np.linalg.norm(Ta_full[inds] - Ta_mechanics[inds]) / np.linalg.norm(Ta_mechanics)
    err_J_TRPN = np.linalg.norm(J_TRPN_full[inds] - J_TRPN_ep[inds]) / np.linalg.norm(J_TRPN_ep)
    ax[2, 0].plot(
        t[inds],
        J_TRPN_ep[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[2, 0].set_ylabel("J TRPN ")

    ax[2, 1].plot(
        t[inds],
        J_TRPN_full[inds] - J_TRPN_ep[inds],
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[2, 1].set_ylabel("J TRPN ep \n error ")

    ax[3, 0].plot(
        t[inds],
        CaTrpn_ep[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
        # marker='.'
    )

    ax[3, 1].plot(
        t[inds],
        CaTrpn_full[inds] - CaTrpn_ep[inds],
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
        # marker='.'
    )
    ax[3, 0].set_ylabel("CaTrpn ep")
    ax[3, 1].set_ylabel("CaTrpn ep  \n error")

    ax[4, 0].plot(
        t[inds],
        TmB_ep[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[4, 0].set_ylabel("TmB ep")

    ax[4, 1].plot(
        t[inds],
        np.around(TmB_full[inds] - TmB_ep[inds], 16),  # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[4, 1].set_ylabel("TmB ep \n error")

    ax[5, 0].plot(
        t[inds],
        XU_ep[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[5, 1].plot(
        t[inds],
        np.around(XU_full[inds] - XU_ep[inds], 16),  # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[5, 0].set_ylabel("XU ep")
    ax[5, 1].set_ylabel("XU ep \n error")

    ax[6, 0].set_ylabel("ep XS")
    ax[6, 1].set_ylabel("XS ep  \n error  \n (missing)")

    ax[6, 0].plot(
        t[inds],
        XS_ep[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[6, 1].plot(
        t[inds],
        XS_full[inds] - XS_ep[inds],
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )

    ax[7, 0].plot(
        t[inds],
        Zetas_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )

    ax[7, 1].plot(
        t[inds],
        Zetas_full[inds] - Zetas_mechanics[inds],
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[7, 0].set_ylabel("Zetas mech")
    ax[7, 1].set_ylabel("Zetas \n mech error")

    ax[8, 0].plot(
        t[inds],
        Ta_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[8, 0].set_ylabel("Ta (kPa)")

    ax[8, 1].plot(
        t[inds],
        Ta_full[inds] - Ta_mechanics[inds],
        # label=f"tol={tol}, perc={perc}%",
        color=col,
        linestyle=ls,
    )
    ax[8, 1].set_ylabel("Ta \n error (kPa)")

    ax[8, 0].set_xlabel("Time (ms)")
    ax[8, 1].set_xlabel("Time (ms)")

    # for axi in ax.flatten():
    # axi.legend()

    # fig.tight_layout()
    if j == len(tols) - 1:
        fig.subplots_adjust(right=0.95)
        lgd = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig("V_and_Ta.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig("V_and_Ta.png")
