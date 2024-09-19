"""Same as 0D but with varying lambda
"""
from pathlib import Path
import gotranx
import numpy as np
import matplotlib.pyplot as plt
import time

save_traces = True
run_full_model = True
from_init_state = True

#init_state_file = "state_200beats_without_twitch_dt0.1.txt"
init_state_file = "state_200beats_twitchTrue_dt0.1.txt"
#TOR mechanics indices for init state file: 
mech_state_indices = [10, 11, 12, 13, 24, 34, 43]

dt = 0.05
simdur = 450 #10 # Simulation duration
t = np.arange(0, simdur, dt)
# Mech step performed every Nth ep step
# Do a set of simulations with various N:
#Ns = np.array([1, 2, 4, 6, 8, 10, 20,30, 40, 42, 43,100, 200, 400, 800])
Ns = np.array([5])

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
#ode = gotranx.load_ode("ORdmm_Land.ode")
ode = gotranx.load_ode("ToRORd_dynCl_endo_caisplit.ode")

mechanics_comp = ode.get_component("mechanics")
mechanics_ode = mechanics_comp.to_ode()

ep_ode = ode - mechanics_comp
#ep_file = Path("ORdmm_Land_ep.py")
ep_file = Path("ToRORd_dynCl_endo_caisplit_ep.py")


# Generate model code from .ode file
rebuild = False
if not ep_file.is_file() or rebuild:
        
    # Generate code for full model. 
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
    #Path("ORdmm_Land_mechanics.py").write_text(code_mechanics)
    Path("ToRORd_dynCl_endo_caisplit_mechanics.py").write_text(code_mechanics)
    #Path("ORdmm_Land.py").write_text(code)
    Path("ToRORd_dynCl_endo_caisplit.py").write_text(code)

# Import ep, mechanics and full model
#import ORdmm_Land_ep
#import ORdmm_Land_mechanics
#import ORdmm_Land

#model = ORdmm_Land.__dict__
#ep_model = ORdmm_Land_ep.__dict__
#mechanics_model = ORdmm_Land_mechanics.__dict__
import ToRORd_dynCl_endo_caisplit_ep
import ToRORd_dynCl_endo_caisplit_mechanics
import ToRORd_dynCl_endo_caisplit

model = ToRORd_dynCl_endo_caisplit.__dict__
ep_model = ToRORd_dynCl_endo_caisplit_ep.__dict__
mechanics_model = ToRORd_dynCl_endo_caisplit_mechanics.__dict__


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

Ca_index = model["state_index"]("cai")
V_index = model["state_index"]("v")
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

# Create arrays to store the results
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))

if run_full_model:
    Ca_full = np.zeros(len(t))
    V_full = np.zeros(len(t))
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



for N in Ns:
    timing_init = time.perf_counter()
    # Get initial values from the EP model
    if from_init_state:
        y_ep = np.delete(np.loadtxt(init_state_file), mech_state_indices)
    else:
        y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"]()
    ep_missing_values = np.zeros(len(ep_ode.missing_variables))
    
    # Get initial values from the mechanics model
    if from_init_state:
        y_mechanics = np.loadtxt(init_state_file)[mech_state_indices]
    else:
        y_mechanics = mechanics_model["init_state_values"]()
    p_mechanics = mechanics_model["init_parameter_values"]()
    #mechanics_missing_values = np.zeros(len(mechanics_ode.missing_variables))
    mechanics_missing_values = np.array([0.0001])  # For cai split, missing variable is cai. Set the initial value instead of setting to zero 

    # Get the initial values from the full model
    if from_init_state:
        y = np.loadtxt(init_state_file)
    else:
        y = model["init_state_values"]()
    p = model["init_parameter_values"]() # Used in lambda update


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
    
    timings_solveloop = []
    timings_ep_steps = []
    timings_mech_steps = []
    for i, ti in enumerate(t):
        timing_loopstart = time.perf_counter()
        # Set initial lambda
        if ti == 0: 
            lmbda_ti = twitch(ti) 
            p[lmbda_index] = lmbda_ti
            p_mechanics[lmbda_index_mechanics] = lmbda_ti
            dLambda = 0
            p[dLambda_index] = dLambda
            p_mechanics[dLambda_index_mechanics] = dLambda

        if run_full_model:
            # Forward step for the full model
            y[:] = fgr(y, ti, dt, p)
            monitor = mon(ti, y, p)
            V_full[i] = y[V_index]
            Ca_full[i] = y[Ca_index]
            J_TRPN_full[i] = monitor[J_TRPN_index]
            XU_full[i] = monitor[XU_index]
            Ta_full[i] = monitor[Ta_index]
            XS_full[i] = y[XS_index]
            CaTrpn_full[i] = y[CaTrpn_index]
            TmB_full[i] = y[TmB_index]
            Zetas_full[i] = y[Zetas_index]
            dLambda_full[i] = p[dLambda_index]
            lmbda_full[i] = p[lmbda_index]

 
        timing_ep_start = time.perf_counter()
        # Forward step for the EP model
        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        timing_ep_end = time.perf_counter()
        timings_ep_steps.append(timing_ep_end-timing_ep_start)

        # Update missing values for the mechanics model
        mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values) # this function just outputs the value of cai straight from y_ep (does not calculate anything)

        if i % N != 0:
            count += 1
            # Lambda still needs to be updated:
            lmbda_ti = twitch(ti+dt)
            p[lmbda_index] = lmbda_ti
            p_mechanics[lmbda_index_mechanics] = lmbda_ti
            dLambda = (lmbda_ti - prev_lmbda)/dt
            p[dLambda_index] = dLambda
            p_mechanics[dLambda_index_mechanics] = dLambda
            prev_ti = ti
            prev_lmbda = lmbda_ti
            timings_solveloop.append(time.perf_counter() - timing_loopstart)
            continue
                    

        # Store the index of the time step where we performed a step
        inds.append(i)

        timing_mech_start = time.perf_counter()
        # Forward step for the mechanics model
        #y_mechanics[:] = fgr_mechanics(
        #    y_mechanics, ti, count * dt, p_mechanics, mechanics_missing_values
        #)
        # For consistency with other models:
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
                
        timing_mech_end = time.perf_counter()
        timings_mech_steps.append(timing_mech_end - timing_mech_start)
    
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

        timings_solveloop.append(time.perf_counter() - timing_loopstart)

    # Plot the results
    perc = 100 * len(inds) / len(t)
    print(f"Solved on {perc}% of the time steps")
    inds = np.array(inds)
    timing_total = time.perf_counter() - timing_init
    
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
        with open(f"CaTrpn_mech_N{N}.txt", "w") as f:
            np.savetxt(f, CaTrpn_mechanics[inds])
        with open(f"J_TRPN_mech_N{N}.txt", "w") as f:
            np.savetxt(f, J_TRPN_mechanics[inds])
        with open(f"XS_mech_N{N}.txt", "w") as f:
            np.savetxt(f, XS_mechanics[inds])
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

    