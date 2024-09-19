from pathlib import Path
import gotranx
import numpy as np
import time


with_twitch = False
from_init_state = True

# Set ep-time step
dt = 0.1 #0.1 #0.05
bcs = 10#00 #1000
beats = 1 #200#00 
init_beats = 0 #200

if from_init_state:
    init_state_file = "state_1beats_twitchTrue_dt0.1.txt"
    #init_state_file = "state_200beats_without_twitch.txt"
    full_init_states = np.loadtxt(init_state_file)

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
    
    dLambda = (lmbda_ti - prev_lmbda)/dt
    p[dLambda_index] = dLambda
    prev_lmbda = lmbda_ti
    return p, prev_lmbda

# Load the model
#ode = gotranx.load_ode("ORdmm_Land.ode")
#ode = gotranx.load_ode("ToRORd_dynCl_endo.ode")
ode = gotranx.load_ode("ToRORd_dynCl_endo_zetasplit.ode")
#file = Path("ToRORd_dynCl_endo.py")
file = Path("ToRORd_dynCl_endo_zetasplit.py")

# Generate model code from .ode file
rebuild = False
if not file.is_file() or rebuild:
        
    # Generate code for full model. 
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
    )
    Path("ToRORd_dynCl_endo_zetasplit.py").write_text(code)

#import ToRORd_dynCl_endo
import ToRORd_dynCl_endo_zetasplit
#import ORdmm_Land
#model = ORdmm_Land.__dict__
#model = ToRORd_dynCl_endo.__dict__
model = ToRORd_dynCl_endo_zetasplit.__dict__

t = np.arange(0, bcs, dt)

if with_twitch:
    with open("lmbda_function.txt", "w") as f:
        np.savetxt(f, twitch(t))

# Forwared generalized rush larsen scheme for the full model
fgr = model["forward_generalized_rush_larsen"]
# Monitor function for the full model
mon = model["monitor_values"]
Ca_index = model["state_index"]("cai")
V_index = model["state_index"]("v")
Ta_index = model["monitor_index"]("Ta")

lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")



Cais = np.zeros(len(t)*beats)
Vs = np.zeros(len(t)*beats)
Tas = np.zeros(len(t)*beats)

# Get the initial values from the full model
if from_init_state:
    y = full_init_states # from init state
else:
    y = model["init_state_values"]()
p = model["init_parameter_values"]() # Used in lambda update

timing_init = time.perf_counter()
for beat in range(beats):    
    
    V_tmp = Vs[beat * len(t) : (beat + 1) * len(t)]
    Cai_tmp = Cais[beat * len(t) : (beat + 1) * len(t)]
    Ta_tmp = Tas[beat * len(t) : (beat + 1) * len(t)]
        
    if with_twitch:
        prev_lmbda = p[lmbda_index]
        p, prev_lmbda = update_lambda_and_dlambda(np.float64(0), prev_lmbda, dt)

    for i, ti in enumerate(t):
        y[:] = fgr(y, ti, dt, p)
        monitor = mon(ti, y, p)
        V_tmp[i] = y[V_index]
        Cai_tmp[i] = y[Ca_index]
        Ta_tmp[i] = monitor[Ta_index]
        
        if with_twitch:            
            p, prev_lmbda = update_lambda_and_dlambda(ti+dt, prev_lmbda, dt)
        print(f"t: {ti:.12}, beat {beat+1}")
        
timing_total = time.perf_counter() - timing_init

# Save states
with open(f"state_{beats+init_beats}beats_twitch{with_twitch}_dt{dt}.txt", "w") as f:        
    np.savetxt(f,y[:])
        
with open(f"timings_full_to_beats{beats+init_beats}_twitch{with_twitch}_dt{dt}", "w") as f:        
    f.write("Total time\n")
    f.write(f"{timing_total}\n")
    
with open(f"V_full_beats{beats+init_beats}_twitch{with_twitch}_dt{dt}.txt", "w") as f:
    np.savetxt(f, Vs)
with open(f"Ta_full_beats{beats+init_beats}_twitch{with_twitch}_dt{dt}.txt", "w") as f:
    np.savetxt(f, Tas)
with open(f"Ca_full_beats{beats+init_beats}_twitch{with_twitch}_dt{dt}.txt", "w") as f:
    np.savetxt(f, Cais)