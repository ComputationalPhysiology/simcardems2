from pathlib import Path
import gotranx
import numpy as np
import time

save_traces = False
run_full_model = False

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


def update_lambda_and_dlambda(t, prev_lmbda, dt):
    lmbda_ti = twitch(t)
    p[lmbda_index] = lmbda_ti
    p_mechanics[lmbda_index_mechanics] = lmbda_ti
    p_ep[lmbda_index_ep] = lmbda_ti

    dLambda = (lmbda_ti - prev_lmbda)/dt
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

    # Generate code for full model. For comparison
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

# Set ep-time step to 0.05 ms
dt = 0.05
simdur = 10 # Simulation duration
t = np.arange(0, simdur, dt)


with open("lmbda_function.txt", "w") as f:
    np.savetxt(f, twitch(t))


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
Ca_index = model["state_index"]("cai")
V_index = model["state_index"]("v")
Ta_index = model["monitor_index"]("Ta")
J_TRPN_index = model["monitor_index"]("J_TRPN")
CaTrpn_index = model["state_index"]("CaTrpn")
TmB_index = model["state_index"]("TmB")
XU_index = model["monitor_index"]("XU")

lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")
XS_index = model["state_index"]("XS")
Zetas_index = model["state_index"]("Zetas")


Ns = np.array([1, 2, 4, 6, 8, 10, 20, 50, 100, 200])

# Create arrays to store the results
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))
CaTrpn_ep = np.zeros(len(t))

if run_full_model:
    Ca_full = np.zeros(len(t))
    V_full = np.zeros(len(t))
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


for N in Ns:
    timing_init = time.perf_counter()
    # Get initial values from the EP model
    y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"]()
    ep_missing_values = np.repeat(0.0001, len(ep_ode.missing_variables))

    # Get initial values from the mechanics model
    y_mechanics = mechanics_model["init_state_values"]()
    p_mechanics = mechanics_model["init_parameter_values"]()
    mechanics_missing_values = np.repeat(0.0001, len(mechanics_ode.missing_variables))


    # Get the initial values from the full model
    y = model["init_state_values"]()
    p = model["init_parameter_values"]() # Used in lambda update

    mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
    ep_missing_values[:] = mv_mechanics(
        0, y_mechanics, p_mechanics, mechanics_missing_values
    )

    # We will store the previous missing values to check for convergence
    prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
    prev_mechanics_missing_values[:] = mechanics_missing_values

    inds = []
    count = 1
    max_count = 10
    prev_lmbda = p[lmbda_index]
    p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(np.float64(0), prev_lmbda, dt)

    timings_solveloop = []
    timings_ep_steps = []
    timings_mech_steps = []
    for i, ti in enumerate(t):
        timing_loopstart = time.perf_counter()

        if run_full_model:
            # Forward step for the full model
            y[:] = fgr(y, ti, dt, p)
            monitor = mon(ti, y, p)
            V_full[i] = y[V_index]
            Ca_full[i] = y[Ca_index]
            J_TRPN_full[i] = monitor[J_TRPN_index]
            Ta_full[i] = monitor[Ta_index]
            XS_full[i] = y[XS_index]
            Zetas_full[i] = y[Zetas_index]
            dLambda_full[i] = p[dLambda_index]
            CaTrpn_full[i] = y[CaTrpn_index]
            TmB_full[i] = y[TmB_index]
            XU_full[i] = monitor[XU_index]

        # Forward step for the EP model (from cai split)
        timing_ep_start = time.perf_counter()
        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        CaTrpn_ep[i] = y_ep[CaTrpn_index_ep]
        monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values)
        TmB_ep[i] = y_ep[TmB_index_ep]
        XU_ep[i] = monitor_ep[XU_index_ep]
        J_TRPN_ep[i] = monitor_ep[J_TRPN_index_ep]
        XS_ep[i] = y_ep[XS_index_ep]

        timing_ep_end = time.perf_counter()
        timings_ep_steps.append(timing_ep_end-timing_ep_start)

        # Update missing values for the mechanics model
        mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values)

        if i % N != 0:
            count += 1
            p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(ti+dt, prev_lmbda, dt)
            timings_solveloop.append(time.perf_counter() - timing_loopstart)
            continue

        # Store the index of the time step where we performed a step
        inds.append(i)

        timing_mech_start = time.perf_counter()
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
        Zetas_mechanics[i] = y_mechanics[Zetas_index_mechanics]
        dLambda_mechanics[i] = p_mechanics[dLambda_index_mechanics]
        lmbda_mechanics[i] = p_mechanics[lmbda_index_mechanics]

        timing_mech_end = time.perf_counter()
        timings_mech_steps.append(timing_mech_end - timing_mech_start)

        p, p_mechanics, p_ep, prev_lmbda = update_lambda_and_dlambda(ti+dt, prev_lmbda, dt)
        # Update missing values for the EP model
        ep_missing_values[:] = mv_mechanics(
            t, y_mechanics, p_mechanics, mechanics_missing_values
        )

        prev_mechanics_missing_values[:] = mechanics_missing_values

        timings_solveloop.append(time.perf_counter() - timing_loopstart)


    timing_total = time.perf_counter() - timing_init
    perc = 100 * len(inds) / len(t)
    print(f"Solved on {perc}% of the time steps")
    inds = np.array(inds)

    with open(f"timings_N{N}", "w") as f:
        f.write("Init time\n")
        f.write(f"{timing_init}\n")
        f.write("Loop total times\n")
        np.savetxt(f, timings_solveloop)
        f.write("Ep steps times\n")
        np.savetxt(f, timings_ep_steps)
        f.write("Mech steps times\n")
        np.savetxt(f, timings_mech_steps)
        f.write("Total time\n")
        f.write(f"{timing_total}\n")

    if save_traces:
        with open(f"V_ep_N{N}.txt", "w") as f:
            np.savetxt(f, V_ep[inds])
        with open(f"Ta_mech_N{N}.txt", "w") as f:
            np.savetxt(f, Ta_mechanics[inds])
        with open(f"Ca_ep_N{N}.txt", "w") as f:
            np.savetxt(f, Ca_ep[inds])
        with open(f"CaTrpn_ep_N{N}.txt", "w") as f:
            np.savetxt(f, CaTrpn_ep[inds])
        with open(f"J_TRPN_ep_N{N}.txt", "w") as f:
            np.savetxt(f, J_TRPN_ep[inds])
        with open(f"XS_ep_N{N}.txt", "w") as f:
            np.savetxt(f, XS_ep[inds])
        with open(f"Zetas_mech_N{N}.txt", "w") as f:
            np.savetxt(f, Zetas_mechanics[inds])

        if run_full_model:
            with open("V_full.txt", "w") as f:
                np.savetxt(f, V_full)
            with open("Ta_full.txt", "w") as f:
                np.savetxt(f, Ta_full)
            with open("Ca_full.txt", "w") as f:
                np.savetxt(f, Ca_full)
            with open("CaTrpn_full.txt", "w") as f:
                np.savetxt(f, CaTrpn_full)
            with open("J_TRPN_full.txt", "w") as f:
                np.savetxt(f, J_TRPN_full)
            with open("XS_full.txt", "w") as f:
                np.savetxt(f, XS_full)
            with open("Zetas_full.txt", "w") as f:
                np.savetxt(f, Zetas_full)
