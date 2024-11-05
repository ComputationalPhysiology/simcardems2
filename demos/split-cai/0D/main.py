"""Same as 0D but with varying lambda
"""
from pathlib import Path
import gotranx
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import sys


def twitch(t, tstart=0.05, ca_ampl=-0.2):
    tau1 = 0.05 * 1000
    tau2 = 0.110 * 1000

    ca_diast = 0.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (
        -1 / (1 - tau2 / tau1)
    )
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1)
        - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca + 1.0


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
dt = 0.1
# Simulate model for 1000 ms
t = np.arange(0, 700, dt)
fig, ax = plt.subplots()
ax.plot(t, twitch(t))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Lambda")
fig.savefig("twitch.png")

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

# Forwared generalized rush larsen scheme for the mechanics model
fgr_mechanics = mechanics_model["forward_generalized_rush_larsen"]
# Monitor function for the mechanics model
mon_mechanics = mechanics_model["monitor_values"]
# Missing values function for the mechanics model
mv_mechanics = mechanics_model["missing_values"]
# Index of the active tension
Ta_index_mechanics = mechanics_model["monitor_index"]("Ta")

CaTrpn_index_mechanics = mechanics_model["state_index"]("CaTrpn")
TmB_index_mechanics = mechanics_model["state_index"]("TmB")

XU_index_mechanics = mechanics_model["monitor_index"]("XU")
J_TRPN_index_mechanics = mechanics_model["monitor_index"]("J_TRPN")
lmbda_index_mechanics = mechanics_model["parameter_index"]("lmbda")
dLambda_index_mechanics = mechanics_model["parameter_index"]("dLambda")
Zetas_index_mechanics = mechanics_model["state_index"]("Zetas")
XS_index_mechanics = mechanics_model["state_index"]("XS")

CaTrpn_index = model["state_index"]("CaTrpn")
TmB_index = model["state_index"]("TmB")

# Forwared generalized rush larsen scheme for the full model
fgr = model["forward_generalized_rush_larsen"]
# Monitor function for the full model
mon = model["monitor_values"]
# Index of the active tension for the full model
Ta_index = model["monitor_index"]("Ta")
# Index of the J_TRPN for the full model

XU_index = model["monitor_index"]("XU")
J_TRPN_index = model["monitor_index"]("J_TRPN")
lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")
XS_index = model["state_index"]("XS")
Zetas_index = model["state_index"]("Zetas")


# Tolerances to test for when to perform steps in the mechanics model
tols = [
    1e-15,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
]
# Colors for the plots
from itertools import cycle

colors = cycle(["r", "g", "b", "c", "m"])
linestyles = cycle(["-", "--", "-.", ":"])



# Create arrays to store the results
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))

CaTrpn_full = np.zeros(len(t))
TmB_full = np.zeros(len(t))
XU_full = np.zeros(len(t))
J_TRPN_full = np.zeros(len(t))
Ta_full = np.zeros(len(t))
dLambda_full = np.zeros(len(t))
XS_full = np.zeros(len(t))
Zetas_full = np.zeros(len(t))
lmbda_full = np.zeros(len(t))

CaTrpn_mechanics = np.zeros(len(t))
TmB_mechanics = np.zeros(len(t))
Ta_mechanics = np.zeros(len(t))
J_TRPN_mechanics = np.zeros(len(t))
XU_mechanics = np.zeros(len(t))
dLambda_mechanics = np.zeros(len(t))
XS_mechanics = np.zeros(len(t))
Zetas_mechanics = np.zeros(len(t))
lmbda_mechanics = np.zeros(len(t))


fig, ax = plt.subplots(9, 2, sharex=True, figsize=(14, 10))
lines = []
labels = []

for j, (col, ls, tol) in enumerate(zip(colors, linestyles, tols)):
    # Get initial values from the EP model
    y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"]()
    ep_missing_values = np.zeros(len(ep_ode.missing_variables))

    # Get initial values from the mechanics model
    y_mechanics = mechanics_model["init_state_values"]()
    p_mechanics = mechanics_model["init_parameter_values"]()
    #mechanics_missing_values = np.zeros(len(mechanics_ode.missing_variables))
    mechanics_missing_values = np.array([0.0001])  # For cai split, missing variable is cai. Set the initial value instead of setting to zero

    # Get the initial values from the full model
    y = model["init_state_values"]()
    p = model["init_parameter_values"]()


    # Get the default values of the missing values
    # A little bit chicken and egg problem here, but in this specific case we know that
    # the mechanics_missing_values is only the calcium concentration, which is a state variable
    # and this doesn't require any additional information to be calculated.
    mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
    ep_missing_values[:] = mv_mechanics(
        0, y_mechanics, p_mechanics, mechanics_missing_values
    )

    # We will store the previous missing values to check for convergence and use for updating
    prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
    prev_mechanics_missing_values[:] = mechanics_missing_values


    inds = []
    count = 1
    max_count = 10
    prev_lmbda = p[lmbda_index]
    prev_ti = 0
    for i, ti in enumerate(t):

        # Set initial lambda
        if ti == 0:
            lmbda_ti = twitch(ti)
            p[lmbda_index] = lmbda_ti
            p_mechanics[lmbda_index_mechanics] = lmbda_ti
            dLambda = 0
            p[dLambda_index] = dLambda
            p_mechanics[dLambda_index_mechanics] = dLambda


        # Forward step for the full model
        y[:] = fgr(y, ti, dt, p)
        monitor = mon(ti, y, p)

        J_TRPN_full[i] = monitor[J_TRPN_index]
        XU_full[i] = monitor[XU_index]
        Ta_full[i] = monitor[Ta_index]
        XS_full[i] = y[XS_index]
        CaTrpn_full[i] = y[CaTrpn_index]
        TmB_full[i] = y[TmB_index]
        Zetas_full[i] = y[Zetas_index]
        dLambda_full[i] = p[dLambda_index]
        lmbda_full[i] = p[lmbda_index]



        # Forward step for the EP model
        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        #monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values) #not currently used

        # Update missing values for the mechanics model
        mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values) # this function just outputs the value of cai straight from y_ep (does not calculate anything)

        # Compute the change in the missing values
        change = np.linalg.norm(
            mechanics_missing_values - prev_mechanics_missing_values
        ) / np.linalg.norm(prev_mechanics_missing_values)

        # Check if the change is small enough to continue to the next time step
        if change < tol:
            count += 1
            # Very small change to just continue to next time step
            if count < max_count:
                # Lambda still needs to be updated:
                lmbda_ti = twitch(ti+dt)
                p[lmbda_index] = lmbda_ti
                p_mechanics[lmbda_index_mechanics] = lmbda_ti
                dLambda = (lmbda_ti - prev_lmbda)/dt
                p[dLambda_index] = dLambda
                p_mechanics[dLambda_index_mechanics] = dLambda
                prev_ti = ti
                prev_lmbda = lmbda_ti
                continue


        # Store the index of the time step where we performed a step
        inds.append(i)


        # Forward step for the mechanics model
        #y_mechanics[:] = fgr_mechanics(
        #    y_mechanics, ti, count * dt, p_mechanics, mechanics_missing_values
        #)
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
        Ta_mechanics[i] = monitor_mechanics[Ta_index_mechanics]
        J_TRPN_mechanics[i] = monitor_mechanics[J_TRPN_index_mechanics]
        XU_mechanics[i] = monitor_mechanics[XU_index_mechanics]
        dLambda_mechanics[i] = p_mechanics[dLambda_index_mechanics]
        Zetas_mechanics[i] = y_mechanics[Zetas_index_mechanics]
        XS_mechanics[i] = y_mechanics[XS_index_mechanics]
        lmbda_mechanics[i] = p_mechanics[lmbda_index_mechanics]
        CaTrpn_mechanics[i] = y_mechanics[CaTrpn_index_mechanics]
        TmB_mechanics[i] = y_mechanics[TmB_index_mechanics]



        # Update lambda
        # Should be done after all calculations except ep_missing, which is used for next ep step
        lmbda_ti = twitch(ti+dt)
        p[lmbda_index] = lmbda_ti
        p_mechanics[lmbda_index_mechanics] = lmbda_ti
        dLambda = (lmbda_ti - prev_lmbda)/dt
        p[dLambda_index] = dLambda
        p_mechanics[dLambda_index_mechanics] = dLambda
        prev_ti = ti
        prev_lmbda = lmbda_ti

        # Update missing values for the EP model # J_TRPN for cai split
        ep_missing_values[:] = mv_mechanics(
            t, y_mechanics, p_mechanics, mechanics_missing_values
        )

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
        #marker='.'
    )


    ax[1, 1].plot(
        t[inds],
        dLambda_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[1, 0].set_ylabel("Lambda \n mech")
    ax[1, 1].set_ylabel("dLambda \n mech")

    err_Ta = np.linalg.norm(Ta_full[inds] - Ta_mechanics[inds]) / np.linalg.norm(
        Ta_mechanics
    )

    ax[2, 0].plot(
        t[inds],
        J_TRPN_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[2, 0].set_ylabel("J TRPN \n mech")


    ax[2, 1].plot(
        t[inds],
        np.around(J_TRPN_full[inds] - J_TRPN_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[2, 1].set_ylabel("J TRPN \n mech error")

    ax[3, 0].plot(
        t[inds],
        CaTrpn_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
        #marker='.'
    )

    ax[3, 1].plot(
        t[inds],
        np.around(CaTrpn_full[inds] - CaTrpn_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
        #marker='.'
    )
    ax[3, 0].set_ylabel("CaTrpn \n mech")
    ax[3, 1].set_ylabel("CaTrpn \n mech error")

    ax[4, 0].plot(
        t[inds],
        TmB_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[4, 0].set_ylabel("TmB mech")


    ax[4, 1].plot(
        t[inds],
        np.around(TmB_full[inds] - TmB_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[4, 1].set_ylabel("TmB \n mech error")




    ax[5, 0].plot(
        t[inds],
        XU_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[5, 1].plot(
        t[inds],
        np.around(XU_full[inds] - XU_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[5, 0].set_ylabel("XU mech")
    ax[5, 1].set_ylabel("XU \n mech error")







    ax[6, 0].plot(
        t[inds],
        XS_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[6, 1].plot(
        t[inds],
        np.around(XS_full[inds] - XS_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )
    ax[6, 0].set_ylabel("XS mech")
    ax[6, 1].set_ylabel("XS \n mech error")

    ax[7, 0].plot(
        t[inds],
        Zetas_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )

    ax[7, 1].plot(
        t[inds],
        np.around(Zetas_full[inds] - Zetas_mechanics[inds], 16), # round to float precision
        # label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
        linestyle=ls,
    )


    ax[7, 0].set_ylabel("Zetas mech")
    ax[7, 1].set_ylabel("Zetas\n mech error")

    ax[8, 0].plot(
        t[inds],
        Ta_mechanics[inds],
        color=col,
        linestyle=ls,  # label=f"tol={tol}"
    )
    ax[8, 0].set_ylabel("Ta (kPa)")

    ax[8, 1].plot(
        t[inds],
        np.around(Ta_full[inds] - Ta_mechanics[inds], 16), # round to float precision
        # label=f"tol={tol}, perc={perc}%",
        color=col,
        linestyle=ls,
    )
    ax[8, 1].set_ylabel("Ta error (kPa)")



    ax[8, 0].set_xlabel("Time (ms)")
    ax[8, 1].set_xlabel("Time (ms)")

    # for axi in ax.flatten():
    # axi.legend()

    if j == len(tols) - 1:
        fig.subplots_adjust(right=0.95)
        lgd = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig("V_and_Ta.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig("V_and_Ta.png")
