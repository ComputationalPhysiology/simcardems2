import logging
import gotranx
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import dolfin
import pulse
import beat
import configparser
import sympy
import os
import argparse
import toml
import ufl_legacy as ufl
import matplotlib.pyplot as plt


from simcardems2 import utils
from simcardems2 import mechanicssolver
from simcardems2 import interpolation
from simcardems2.land import LandModel


try:
    raise ImportError
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


logging.getLogger("beat").setLevel(logging.ERROR)


def parse_parameters(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simcardems CLI")
    parser.add_argument("config-file", type=Path, help="Config file")

    args = vars(parser.parse_args(argv))
    try:
        config = toml.loads(args["config-file"].read_text())
    except toml.TomlDecodeError as e:
        print(f"Error when parsing parameters. Check that all parameters are defined as the correct types in config file. Error: {e}")
        exit(1)
    return config

config = parse_parameters()

def validate_input(config):    
    number_par = ["sim_dur", 
                  "dt", 
                  "N", 
                  "sigma_il", 
                  "sigma_it", 
                  "sigma_el", 
                  "sigma_et", 
                  "mech_a", 
                  "mech_a_f",
                  "mech_b", 
                  "mech_b_f", 
                  "mech_a_s", 
                  "mech_b_s", 
                  "mech_a_fs", 
                  "mech_b_fs"]
    
    for param in number_par:
        if not isinstance(config[param], (int, float)):
            raise ValueError(f"Parameter '{param} = {config[param]}' is a {type(config[param]).__name__}. Provide an integer or a float")
    
    int_par = ["N"]
    for param in int_par:
        if not isinstance(config[param], (int)):
            raise ValueError(f"Parameter '{param} = {config[param]}' is a {type(config[param]).__name__}. Provide an integer")



validate_input(config)

# TODO: update to use Neumann BC from config

stim_region = config["stim"]["xlim"], config["stim"]["ylim"], config["stim"]["zlim"]
outdir = Path(config["outdir"])
outdir.mkdir(parents=True, exist_ok=True)

with open(Path(outdir / "config.txt"), "w") as f:
    f.write(toml.dumps(config))
    
print(config)

#for bc in range(config['bcs']['numbers']):
#    bcs_markers[f"{bc}"] = int(config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].marker'))
#    bcs_types[f"{bc}"] = config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].type')
#    bcs_variables[f"{bc}"] = config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].variable')
    
#    bcs_val = config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].func')
    
    # Try converting bcs to float, otherwise treat as function
#    try:
#        bcs_values[f"{bc}"] = dolfin.Constant(float(bcs_val))
#    except ValueError:
#        print(f"Function bcs: {bcs_val}")
        #bcs_values[f"{bc}"] = dolfin.Expression('alpha*t', alpha=traction_alpha, t=t_bcs, degree=0)
        
        
        # TODO: update so you can use functions as bcs
        
        #bcs_func_params = config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].func_params').split(',')
    
        #parameters_dict = {str(param): 0 for param in bcs_func_params}
        #bcs_values[f"{bc}"] = dolfin.Expression(bcs_val,**parameters_dict, t=t_bcs)
    
        #for i, param in enumerate(bcs_func_params):
        #    print(i, param)
        #    param_value = config.get('BOUNDARY CONDITIONS', f'bcs[{bc}].func_params.{param}')
        #    bcs_values[f"{bc}"].user_parameters[param] = param_value
            





# TODO: do a different check than this later. something smoother
bcs_dirichlet = []
bcs_neumann = []
for bc in range(config['bcs']['numbers']):
    if config['bcs'][f"{bc}"]['type'] == 'Dirichlet':
        bcs_dirichlet.append(bc)
    elif config['bcs'][f"{bc}"]['type'] =='Neumann':
        bcs_neumann.append(bc)
    else:
        raise KeyError(f'{config["bcs"][f"{bc}"]["type"]} is not a valid type of boundary condition. Use Dirichlet or Neumann. Check config file')

        

print(f'Ep variables to output: {config["out_ep"]}')
print(f'Mech variables to output: {config["out_mech"]}')

mesh = dolfin.Mesh()
with dolfin.XDMFFile(f'{config["mech_mesh"]}.xdmf') as infile:
    infile.read(mesh)
print(f'Loaded mesh: {config["mech_mesh"]}')


ffun_bcs = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with dolfin.XDMFFile(f'{config["bcs"]["markerfile"]}.xdmf') as infile:
    infile.read(ffun_bcs)
print(f'Loaded markerfile for bcs: {config["bcs"]["markerfile"]}')


tol = 5e-4
# Surface to volume ratio
chi = 140.0  # mm^{-1}
# Membrane capacitance
C_m = 0.01  # mu F / mm^2
cm2mm = 10.0

t = np.arange(0, config["sim_dur"], config["dt"]) 
sigma = [config["sigma_il"], config["sigma_it"], config["sigma_el"], config["sigma_et"]] 
material_parameters = dict(a=config["mech_a"],a_f=config["mech_a_f"],b=config["mech_b"],b_f=config["mech_b_f"],a_s=config["mech_a_s"],b_s=config["mech_b_s"],a_fs=config["mech_a_fs"],b_fs=config["mech_b_fs"])


dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4


def define_conductivity_tensor(sigma, chi, C_m):
    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma[0], sigma[2])
    sigma_t = harmonic_mean(sigma[1], sigma[3])

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    M = dolfin.as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M


def define_stimulus(mesh, chi, C_m, time, stim_region=stim_region,  stim_start = config["stim"]["start"], A=config["stim"]["amplitude"], duration=config["stim"]["duration"]):
    S1_marker = 1
    S1_subdomain = dolfin.CompiledSubDomain(
        ' '.join((f'x[0] >= {stim_region[0][0]}',
                  f'&& x[0] <= {stim_region[0][1]}',
                  f'&& x[1] >= {stim_region[1][0]}',
                  f'&& x[1] <= {stim_region[1][1]}',
                  f'&& x[2] >= {stim_region[2][0]}',
                  f'&& x[2] <= {stim_region[2][1]}',
                  ))
        )
    
    S1_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)
    with dolfin.XDMFFile((outdir / "stim_region_markers.xdmf").as_posix()) as xdmf:
        xdmf.write(S1_markers)

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms

    print(f"duration: {duration}, amplitude: {amplitude}")
    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=config["stim"]["start"],
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )

    dx = dolfin.Measure("dx", domain=mesh, subdomain_data=S1_markers)(S1_marker)
    return beat.base_model.Stimulus(dz=dx, expr=I_s)



# Load the model
if not Path("ep_model.py").exists():
    ode = gotranx.load_ode(config["modelfile"])

    mechanics_comp = ode.get_component("mechanics")
    mechanics_ode = mechanics_comp.to_ode()

    ep_ode = ode - mechanics_comp

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

    Path("ep_model.py").write_text(code_ep)
    Path("mechanics_model.py").write_text(code_mechanics)

import ep_model as _ep_model
ep_model = _ep_model.__dict__

# Get index of ep state-parameters to output
out_indices = {} 
for out_ep_var in config["out_ep"]:
    try:
        out_indices[out_ep_var] = ep_model["state_index"](out_ep_var)
    except KeyError:
        print(f"{out_ep_var} is not a valid ep variable. Check config file")
    
# Forwared generalized rush larsen scheme for the electrophysiology model
fgr_ep = jit(nopython=True)(ep_model["forward_generalized_rush_larsen"])
# Monitor function for the electrophysiology model
mon_ep = ep_model["monitor_values"]
# Missing values function for the electrophysiology model
mv_ep = ep_model["missing_values"]
lmbda_index_ep = ep_model["parameter_index"]("lmbda")



# Get initial values from the EP model
y_ep_ = ep_model["init_state_values"]()
p_ep_ = ep_model["init_parameter_values"](amp=0.0)

ep_missing_values_ = np.zeros(len(ep_model["missing"]))
ep_mesh = dolfin.adapt(dolfin.adapt(dolfin.adapt(mesh)))

time = dolfin.Constant(0.0)
I_s = define_stimulus(mesh=ep_mesh, 
                      chi=chi, 
                      C_m=C_m, 
                      time=time, 
                      stim_region=stim_region,
                      stim_start=config["stim"]["start"],
                      A=config["stim"]["amplitude"], 
                      duration=config["stim"]["duration"])
M = define_conductivity_tensor(sigma, chi, C_m)
params = {"preconditioner": "sor", "use_custom_preconditioner": False}
ep_ode_space = dolfin.FunctionSpace(ep_mesh, "CG", 1)
v_ode = dolfin.Function(ep_ode_space)
num_points_ep = v_ode.vector().local_size()
lmbda = dolfin.Function(ep_ode_space)

y_ep = np.zeros((len(y_ep_), num_points_ep))
y_ep.T[:] = y_ep_

mechanics_missing_values_ = np.zeros(2)

# Set the activation
activation_space = dolfin.FunctionSpace(mesh, "CG", 1)
activation = dolfin.Function(activation_space)
num_points_mech = activation.vector().local_size()


missing_mech = interpolation.MissingValue(
    element=activation.ufl_element(),
    interpolation_element=ep_ode_space.ufl_element(),
    mechanics_mesh=mesh,
    ep_mesh=ep_mesh,
    num_values=len(mechanics_missing_values_),
)

missing_ep = interpolation.MissingValue(
    element=ep_ode_space.ufl_element(),
    interpolation_element=activation.ufl_element(),
    mechanics_mesh=mesh,
    ep_mesh=ep_mesh,
    num_values=len(ep_missing_values_),
)


missing_mech.values_ep.T[:] = mechanics_missing_values_
missing_ep.values_mechanics.T[:] = ep_missing_values_
missing_ep.values_ep.T[:] = ep_missing_values_
missing_mech.values_mechanics.T[:] = mechanics_missing_values_


# Create function spaces for ep state parameters to output
out_ep_funcs = {}
for out_ep_var in config["out_ep"]:
    out_ep_funcs[out_ep_var] = dolfin.Function(ep_ode_space)
    

p_ep = np.zeros((len(p_ep_), num_points_ep))
p_ep.T[:] = p_ep_

pde = beat.MonodomainModel(time=time, mesh=ep_mesh, M=M, I_s=I_s, params=params)
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfin.Function(ep_ode_space),
    v_pde=pde.state,
    fun=fgr_ep,
    parameters=p_ep,
    init_states=y_ep,
    num_states=len(y_ep),
    v_index=ep_model["state_index"]("v"),
    missing_variables=missing_ep.values_ep,
    num_missing_variables=len(ep_model["missing"]),
)

ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=0.5)

marker_functions = pulse.MarkerFunctions(ffun=ffun_bcs)

def create_boundary_conditions(ffun_bcs, bcs_dirichlet, bcs_neumann, bcs_dict):  # TODO: update to not need separate dirichlet and neumann list 
    def dirichlet_bc(W):
        bcs_W = {
            'u_x' : W.sub(0).sub(0),
            'u_y' : W.sub(0).sub(1),
            'u_z' : W.sub(0).sub(2),
            # TODO: add the rest (check dolfin doc)
            } 
        
        bcs = []
        for bc in bcs_dirichlet:
            bcs.append(
                dolfin.DirichletBC(
                    bcs_W[bcs_dict[f"{bc}"]["variable"]],
                    bcs_dict[f"{bc}"]["func"],
                    ffun_bcs,
                    bcs_dict[f"{bc}"]["marker"]
                ))
        return bcs
       
    neumann_bc = []
    # TODO: add support for using neumann 
    #if bcs_neumann is not None:
    #    for bc in bcs_neumann:
    #        neumann_bc.append(
    #            pulse.NeumannBC(
    #                traction=utils.float_to_constant(bcs_values[f"{bc}"]),
    #                marker=bcs_markers[f"{bc}"],
    #            ))
        
    # Collect Boundary Conditions
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,),neumann=neumann_bc)
    return bcs


f0 = dolfin.as_vector([1.0, 0.0, 0.0])
s0 = dolfin.as_vector([0.0, 1.0, 0.0])
n0 = dolfin.as_vector([0.0, 0.0, 1.0])
microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)
geometry = pulse.Geometry(
    mesh=mesh, marker_functions=marker_functions, microstructure=microstructure
)

active_model = LandModel(
    f0=f0,
    s0=s0,
    n0=n0,
    XS=missing_mech.u_mechanics[0],
    XW=missing_mech.u_mechanics[1],
    mesh=mesh,
    eta=0,
    dLambda_tol=1e-12,
)
active_model.t = 0.0


mech_variables = {
    'Ta': active_model.Ta_current, 
    'Zetas': active_model.Zetas, 
    'Zetaw': active_model.Zetaw,
    'lambda': active_model.lmbda,
    }

for out_mech_var in config["out_mech"]:
    assert out_mech_var in mech_variables, f"Error: '{out_mech_var}' is not a valid variable name. Check config file" 

material = pulse.HolzapfelOgden(
    parameters=material_parameters,
    active_model=active_model,
    f0=f0,
    s0=s0,
    n0=n0,
)


def compute_function_average_over_mesh(func, mesh):
    volume = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(domain=mesh))
    return dolfin.assemble(func * dolfin.dx(domain=mesh)) / volume

# Collect Boundary Conditions
bcs = create_boundary_conditions(ffun_bcs, bcs_dirichlet, bcs_neumann, config['bcs'])

problem = mechanicssolver.MechanicsProblem(geometry, material, bcs)
problem.solve(0.0, 0.0)

disp_file = Path(outdir / "disp.xdmf")
disp_file.unlink(missing_ok=True)
disp_file.with_suffix(".h5").unlink(missing_ok=True)

out_ep_files = {}
for out_ep_var in config["out_ep"]:
    out_ep_files[out_ep_var] = Path(outdir /f"{out_ep_var}_out_ep.xdmf")
    out_ep_files[out_ep_var].unlink(missing_ok=True)
    out_ep_files[out_ep_var].with_suffix(".h5").unlink(missing_ok=True)

out_mech_files = {}
for out_mech_var in config["out_mech"]:
    out_mech_files[out_mech_var] = Path(outdir /f"{out_mech_var}_out_mech.xdmf")
    out_mech_files[out_mech_var].unlink(missing_ok=True)
    out_mech_files[out_mech_var].with_suffix(".h5").unlink(missing_ok=True)


# Create arrays for storing values to plot time series for example nodes
out_ep_example_nodes = {}
for out_ep_var in config["out_ep"]:
    out_ep_example_nodes[out_ep_var] = np.zeros(len(t))

out_mech_example_nodes = {}
for out_mech_var in config["out_mech"]:
    out_mech_example_nodes[out_mech_var] = np.zeros(len(t))

# Create arrays for storing values to plot time series for volume averages 
out_ep_volume_average_timeseries = {}
for out_ep_var in config["out_ep"]:
    out_ep_volume_average_timeseries[out_ep_var] = np.zeros(len(t))

out_mech_volume_average_timeseries = {}
for out_mech_var in config["out_mech"]:
    out_mech_volume_average_timeseries[out_mech_var] = np.zeros(len(t))
    
        

inds = [] # Array with time-steps for which we solve mechanics
j = 0
theta = 0.5
timer = dolfin.Timer("solve_loop")
for i, ti in enumerate(t):
    print(f"Solving time {ti:.2f} ms")
    #t_bcs.assign(ti)
    ep_solver.step((ti, ti + config["dt"]))

    # Assign values to ep function
    for out_ep_var in config["out_ep"]:
        out_ep_funcs[out_ep_var].vector()[:] = ode._values[out_indices[out_ep_var]]
        
    # Store values to plot time series for given coord
    for out_ep_var in config["out_ep_point"]["variables"]:
        out_ep_example_nodes[out_ep_var][i] = out_ep_funcs[out_ep_var](config["out_ep_point"][f"{out_ep_var}"])
        
    for out_mech_var in config["out_mech_point"]["variables"]:
        out_mech_example_nodes[out_mech_var][i] = mech_variables[out_mech_var](config["out_mech_point"][f"{out_mech_var}"])
        
    # Compute volume averages for selected parameters
    for out_ep_var in config["out_ep"]:
        out_ep_volume_average_timeseries[out_ep_var][i] = compute_function_average_over_mesh(out_ep_funcs[out_ep_var], ep_mesh)
        
    for out_mech_var in config["out_mech"]:
        out_mech_volume_average_timeseries[out_mech_var][i] = compute_function_average_over_mesh(mech_variables[out_mech_var], mesh)


    if i % config["N"] != 0:
        continue
    missing_ep_values = mv_ep(
        ti + config["dt"], ode._values, ode.parameters, missing_ep.values_ep
    )

    for k in range(missing_mech.num_values):
        missing_mech.u_ep_int[k].vector()[:] = missing_ep_values[k, :]

    missing_mech.interpolate_ep_to_mechanics()
    missing_mech.mechanics_function_to_values()
    inds.append(i)

    print("Solve mechanics")
    problem.solve(ti, config["N"] * config["dt"])
    active_model.update_prev()

    missing_ep.u_mechanics_int[0].interpolate(active_model.Zetas)
    missing_ep.u_mechanics_int[1].interpolate(active_model.Zetaw)

    missing_ep.interpolate_mechanics_to_ep()
    missing_ep.ep_function_to_values()
    lmbda.interpolate(active_model.lmbda)
    p_ep[lmbda_index_ep, :] = lmbda.vector().get_local()
    print(
        active_model.lmbda.vector().get_local().min(),
        active_model.lmbda.vector().get_local().max(),
    )

    U, p = problem.state.split(deepcopy=True)
    

    with dolfin.XDMFFile(disp_file.as_posix()) as file:
        file.write_checkpoint(U, "disp", j, dolfin.XDMFFile.Encoding.HDF5, True)  
    for out_ep_var in config["out_ep"]:
        with dolfin.XDMFFile(out_ep_files[out_ep_var].as_posix()) as file:
            file.write_checkpoint(out_ep_funcs[out_ep_var], out_ep_var, j, dolfin.XDMFFile.Encoding.HDF5, True)
    for out_mech_var in config["out_mech"]:
        with dolfin.XDMFFile(out_mech_files[out_mech_var].as_posix()) as file:
            file.write_checkpoint(mech_variables[out_mech_var], out_mech_var, j, dolfin.XDMFFile.Encoding.HDF5, True)
    
    
    j += 1
timer.stop()
timings = dolfin.timings(
    dolfin.TimingClear.keep,
    [dolfin.TimingType.wall, dolfin.TimingType.user, dolfin.TimingType.system],
).str(True)
print(timings)
with open(Path(outdir / "solve_timings.txt"), "w") as f:
    f.write(timings)

# Write averaged results for later analysis    
for out_ep_var in config["out_ep"]:
    with open(Path(outdir / f"{out_ep_var}_out_ep_volume_average.txt"), "w") as f:
        np.savetxt(f, out_ep_volume_average_timeseries[out_ep_var][inds])
        
for out_mech_var in config["out_mech"]:
    with open(Path(outdir / f"{out_mech_var}_out_mech_volume_average.txt"), "w") as f:
        np.savetxt(f, out_mech_volume_average_timeseries[out_mech_var][inds])
    

print(f"Solved on {100 * len(inds) / len(t)}% of the time steps")
inds = np.array(inds)

# Plot the results
fig, ax = plt.subplots(len(config["out_ep"]),1, figsize=(10, 10))
if len(config["out_ep"]) == 1:
    ax = np.array([ax])
for i, out_ep_var in enumerate(config["out_ep"]):
    ax[i].plot(t[inds], out_ep_volume_average_timeseries[out_ep_var][inds])
    ax[i].set_title(f"{out_ep_var} volume average")
    ax[i].set_xlabel("Time (ms)")
fig.tight_layout()
fig.savefig(Path(outdir /"out_ep_volume_averages.png") )
    
fig, ax = plt.subplots(len(config["out_ep_point"]["variables"]),1, figsize=(10, 10))
if len(config["out_ep_point"]["variables"]) == 1:
    ax = np.array([ax])
for i, out_ep_var in enumerate(config["out_ep_point"]["variables"]):
    ax[i].plot(t[inds], out_ep_example_nodes[out_ep_var][inds])
    ax[i].set_title(f'{out_ep_var} in coord {config["out_ep_point"][f"{out_ep_var}"]}')
    ax[i].set_title(f'{out_ep_var} in coord {config["out_ep_point"][f"{out_ep_var}"]}')
    ax[i].set_xlabel('Time (ms)')
fig.tight_layout()
fig.savefig(Path(outdir /"out_ep_coord.png") )

fig, ax = plt.subplots(len(config["out_mech"]),1, figsize=(10, 10))
if len(config["out_mech"]) == 1:
    ax = np.array([ax])
for i, out_mech_var in enumerate(config["out_mech"]):
    ax[i].plot(t[inds], out_mech_volume_average_timeseries[out_mech_var][inds])
    ax[i].set_title(f"{out_mech_var} volume average")
    ax[i].set_xlabel("Time (ms)")
fig.tight_layout()
fig.savefig(Path(outdir /"out_mech_volume_averages.png") ) 
    
fig, ax = plt.subplots(len(config["out_mech_point"]["variables"]),1, figsize=(10, 10))
if len(config["out_mech_point"]["variables"]) == 1:
    ax = np.array([ax])
for i, out_mech_var in enumerate(config["out_mech_point"]["variables"]):
    ax[i].plot(t[inds], out_mech_example_nodes[out_mech_var][inds])
    ax[i].set_title(f'{out_mech_var} in coord {config["out_mech_point"][f"{out_mech_var}"]}')
    ax[i].set_xlabel('Time (ms)')
fig.tight_layout()
fig.savefig(Path(outdir /"out_mech_coord.png") )

