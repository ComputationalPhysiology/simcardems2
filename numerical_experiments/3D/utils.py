from dataclasses import dataclass, field
from copy import deepcopy
import logging
from typing import Sequence, Any
import argparse
from pathlib import Path
import toml
import json
import numpy as np
import gotranx
import dolfin
import beat
import pulse
import matplotlib.pyplot as plt
import simcardems2
from simcardems2 import mechanicssolver

try:
    raise ImportError
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


cm2mm = 10.0


def run_3D(config: dict[str, Any] | None = None):
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    logging.getLogger("beat").setLevel(logging.ERROR)

    config = parse_parameters(config=config)

    mesh, ep_mesh = read_mesh(config)

    # Load the model
    ep_model = setup_ep_ode_model(config["sim"]["modelfile"], label=config["sim"]["split_scheme"])

    # Forwared generalized rush larsen scheme for the electrophysiology model
    fgr_ep = jit(nopython=True)(ep_model["forward_generalized_rush_larsen"])

    # Missing values function for the electrophysiology model
    mv_ep = ep_model["missing_values"]

    # Get initial values from the EP model
    y_ep_ = ep_model["init_state_values"]()
    p_ep_ = ep_model["init_parameter_values"](i_Stim_Amplitude=0.0)

    if config["sim"]["split_scheme"] == "cai":
        # Init value for J_TRPN = dCaTrpn_dt * trpnmax (to compare with zetasplit)
        ep_missing_values_ = np.array([0.00010730972184715098])
        # Init value for cai
        mechanics_missing_values_ = np.array([0.0001])
    elif config["sim"]["split_scheme"] == "zeta":
        ep_missing_values_ = np.zeros(len(ep_model["missing"]))
        mechanics_missing_values_ = np.zeros(2)
    elif config["sim"]["split_scheme"] == "cai_catrpn":
        ep_missing_values_ = np.array([])
        # Init value for cai
        mechanics_missing_values_ = np.array([0.0001])
    else:
        raise ValueError(f"Unknown split scheme: {config['sim']['split_scheme']}")

    ep_ode_space = dolfin.FunctionSpace(ep_mesh, "DG", 1)
    v_ode = dolfin.Function(ep_ode_space)
    num_points_ep = v_ode.vector().local_size()

    y_ep = np.zeros((len(y_ep_), num_points_ep))
    y_ep.T[:] = y_ep_  # Set to y_ep with initial values defined in ep_model

    # Set the activation
    activation_space = dolfin.FunctionSpace(mesh, "DG", 1)
    activation = dolfin.Function(activation_space)

    missing_mech = simcardems2.interpolation.MissingValue(
        element=activation.ufl_element(),
        interpolation_element=ep_ode_space.ufl_element(),
        mechanics_mesh=mesh,
        ep_mesh=ep_mesh,
        num_values=len(mechanics_missing_values_),
    )

    if len(ep_missing_values_) > 0:
        missing_ep = simcardems2.interpolation.MissingValue(
            element=ep_ode_space.ufl_element(),
            interpolation_element=activation.ufl_element(),
            mechanics_mesh=mesh,
            ep_mesh=ep_mesh,
            num_values=len(ep_missing_values_),
        )

        missing_ep.values_mechanics.T[:] = ep_missing_values_
        missing_ep.values_ep.T[:] = ep_missing_values_
        ode_missing_variables = missing_ep.values_ep
        missing_ep_args = (missing_ep.values_ep,)
    else:
        missing_ep = None
        ode_missing_variables = None
        missing_ep_args = ()

    missing_mech.values_ep.T[:] = mechanics_missing_values_
    missing_mech.values_mechanics.T[:] = mechanics_missing_values_
    missing_mech.mechanics_values_to_function()  # Assign initial values to mech functions

    # Use previous cai in mech to be consistent across splitting schemes
    prev_missing_mech = simcardems2.interpolation.MissingValue(
        element=activation.ufl_element(),
        interpolation_element=ep_ode_space.ufl_element(),
        mechanics_mesh=mesh,
        ep_mesh=ep_mesh,
        num_values=len(mechanics_missing_values_),
    )
    for i in range(len(mechanics_missing_values_)):
        prev_missing_mech.u_mechanics[i].vector().set_local(missing_mech.values_mechanics[i])

    p_ep = np.zeros((len(p_ep_), num_points_ep))
    p_ep.T[:] = p_ep_  # Initialise p_ep with initial values defined in ep_model

    pde = setup_monodomain_model(config, ep_mesh)

    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfin.Function(ep_ode_space),
        v_pde=pde.state,
        fun=fgr_ep,
        parameters=p_ep,
        init_states=y_ep,
        num_states=len(y_ep),
        v_index=ep_model["state_index"]("v"),
        missing_variables=ode_missing_variables,
        num_missing_variables=len(ep_missing_values_),
    )
    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # TODO: do a different dirichlet/neumann check than this later. something smoother
    t_bcs = dolfin.Constant(0)

    if config["sim"]["split_scheme"] == "cai":
        land_params = {"cai": mechanics_missing_values_[0]}
        LandModel = simcardems2.land_caisplit.LandModel

    elif config["sim"]["split_scheme"] == "zeta":
        land_params = {
            "XS": mechanics_missing_values_[0],
            "XW": mechanics_missing_values_[1],
        }
        LandModel = simcardems2.land_Zetasplit.LandModel
    elif config["sim"]["split_scheme"] == "cai_catrpn":
        land_params = {"CaTrpn": mechanics_missing_values_[0]}
        LandModel = simcardems2.land_CaTrpnsplit.LandModel
    else:
        raise ValueError(f"Unknown split scheme: {config['sim']['split_scheme']}")

    problem = setup_mechancis_model(
        config,
        mesh,
        LandModel,
        land_params=land_params,
        t_bcs=t_bcs,
    )

    active_model = problem.material.active

    if config["sim"]["split_scheme"] == "cai":
        mech_variables = {
            "Ta": active_model.Ta_current,
            "Zetas": active_model._Zetas,
            "Zetaw": active_model._Zetaw,
            "lambda": active_model.lmbda,
            "XS": active_model._XS,
            "XW": active_model._XW,
            "TmB": active_model._TmB,
            "CaTrpn": active_model._CaTrpn,
            "J_TRPN": active_model._J_TRPN,
        }
    elif config["sim"]["split_scheme"] == "zeta":
        mech_variables = {
            "Ta": active_model.Ta_current,
            "Zetas": active_model._Zetas,
            "Zetaw": active_model._Zetaw,
            "lambda": active_model.lmbda,
            "XS": active_model.XS,
            "XW": active_model.XW,
            "dLambda": active_model._dLambda,
        }
    else:
        mech_variables = {
            "Ta": active_model.Ta_current,
            "Zetas": active_model._Zetas,
            "Zetaw": active_model._Zetaw,
            "lambda": active_model.lmbda,
            "XS": active_model._XS,
            "XW": active_model._XW,
            "TmB": active_model._TmB,
        }

    collector = DataCollector(
        problem=problem,
        ep_ode_space=ep_ode_space,
        config=config,
        mech_variables=mech_variables,
    )

    inds = []  # Array with time-steps for which we solve mechanics
    j = 0

    for i, ti in enumerate(collector.t):
        collector.timers.start_single_loop()

        print(f"Solving time {ti:.2f} ms")
        t_bcs.assign(ti)  # Use ti+ dt here instead?

        collector.timers.start_ep()
        ep_solver.step((ti, ti + config["sim"]["dt"]))
        collector.timers.stop_ep()

        # Assign values to ep function
        for out_ep_var in collector.out_ep_names:
            collector.out_ep_funcs[out_ep_var].vector()[:] = ode._values[
                ep_model["state_index"](out_ep_var)
            ]

        collector.write_node_data_ep(i)

        if i % config["sim"]["N"] != 0:
            collector.timers.stop_single_loop()
            continue

        collector.timers.start_var_transfer()
        # Extract missing values for the mechanics step from the ep model (ep function space)
        missing_ep_values = mv_ep(
            ti + config["sim"]["dt"],
            ode._values,
            ode.parameters,
            *missing_ep_args,
        )
        # Assign the extracted values as missing_mech for the mech step (ep function space)
        for k in range(missing_mech.num_values):
            missing_mech.u_ep_int[k].vector()[:] = missing_ep_values[k, :]

        # Interpolate missing variables from ep to mech function space
        missing_mech.interpolate_ep_to_mechanics()
        missing_mech.mechanics_function_to_values()
        inds.append(i)

        collector.timers.stop_var_transfer()

        print("Solve mechanics")
        collector.timers.start_mech()

        active_model.t.assign(ti + config["sim"]["N"] * config["sim"]["dt"])  # Addition!
        nit, conv = problem.solve(ti, config["sim"]["N"] * config["sim"]["dt"])
        collector.timers.no_of_newton_iterations.append(nit)
        print(f"No of iterations: {nit}")
        active_model.update_prev()
        collector.timers.stop_mech()

        collector.timers.start_var_transfer()
        # Do we need to handle more cases here?
        if config["sim"]["split_scheme"] == "cai":
            missing_ep.u_mechanics_int[0].interpolate(active_model._J_TRPN)
        if missing_ep is not None:
            missing_ep.interpolate_mechanics_to_ep()
            missing_ep.ep_function_to_values()
        collector.timers.stop_var_transfer()

        collector.write_node_data_mech(i)

        collector.timers.start_var_transfer
        # Use previous cai in mech to be consistent with zeta split
        for i in range(len(mechanics_missing_values_)):
            prev_missing_mech.u_mechanics[i].vector().set_local(missing_mech.values_mechanics[i])
        collector.timers.stop_var_transfer()
        collector.timers.collect_var_transfer()

        collector.write_disp(j)

        j += 1
        collector.timers.stop_single_loop()

    collector.finalize(inds)


def default_config():
    return {
        "ep": {
            "conductivities": {
                "sigma_el": 0.62,
                "sigma_et": 0.24,
                "sigma_il": 0.17,
                "sigma_it": 0.019,
            },
            "stimulus": {
                "amplitude": 50000.0,
                "duration": 2,
                "start": 0.0,
                "xmax": 1.5,
                "xmin": 0.0,
                "ymax": 1.5,
                "ymin": 0.0,
                "zmax": 1.5,
                "zmin": 0.0,
            },
            "chi": 140.0,
            "C_m": 0.01,
        },
        "mechanics": {
            "material": {
                "a": 2.28,
                "a_f": 1.686,
                "a_fs": 0.0,
                "a_s": 0.0,
                "b": 9.726,
                "b_f": 15.779,
                "b_fs": 0.0,
                "b_s": 0.0,
            },
            "bcs": [
                {"V": "u_x", "expression": 0, "marker": 1, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_y", "expression": 0, "marker": 3, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_z", "expression": 0, "marker": 5, "param_numbers": 0, "type": "Dirichlet"},
            ],
        },
        "sim": {
            "N": 2,
            "dt": 0.05,
            "mech_mesh": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz",
            "markerfile": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz_surface_ffun",
            "modelfile": "../odefiles/ToRORd_dynCl_endo_caisplit.ode",
            "outdir": "100ms_N1_cai_split_runcheck",
            "sim_dur": 4,
            "split_scheme": "cai",
        },
        "output": {
            "all_ep": ["v"],
            "all_mech": ["Ta", "lambda"],
            "point_ep": [
                {"name": "v", "x": 0, "y": 0, "z": 0},
            ],
            "point_mech": [
                {"name": "Ta", "x": 0, "y": 0, "z": 0},
                {"name": "lambda", "x": 0, "y": 0, "z": 0},
            ],
        },
    }


def deep_update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_parameters(
    argv: Sequence[str] | None = None, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    _config = default_config()

    if config is None:
        # Parse config file from the command line
        parser = argparse.ArgumentParser(description="Simcardems CLI")
        parser.add_argument("config-file", type=Path, help="Config file")

        args = vars(parser.parse_args(argv))
        try:
            config = toml.loads(args["config-file"].read_text())
        except toml.TomlDecodeError as e:
            print(f"Error when parsing input parameters. Check config file. Error: {e}")
            exit(1)

    return deep_update_dict(_config, config)


def setup_ep_ode_model(odefile, label):
    module_file = Path(f"ep_model_{label}.py")
    if not module_file.is_file():
        ode = gotranx.load_ode(odefile)

        mechanics_comp = ode.get_component("mechanics")
        mechanics_ode = mechanics_comp.to_ode()

        ep_ode = ode - mechanics_comp

        # Generate code for the electrophysiology model
        code_ep = gotranx.cli.gotran2py.get_code(
            ep_ode,
            scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
            missing_values=mechanics_ode.missing_variables,
        )

        Path(module_file).write_text(code_ep)
        # Currently 3D mech needs to be written manually

    return __import__(str(module_file.stem)).__dict__


def define_stimulus(mesh, chi, C_m, time, config):
    stim_region = (
        [config["ep"]["stimulus"]["xmin"], config["ep"]["stimulus"]["xmax"]],
        [config["ep"]["stimulus"]["ymin"], config["ep"]["stimulus"]["ymax"]],
        [config["ep"]["stimulus"]["zmin"], config["ep"]["stimulus"]["zmax"]],
    )
    stim_start = config["ep"]["stimulus"]["start"]
    A = config["ep"]["stimulus"]["amplitude"]
    duration = config["ep"]["stimulus"]["duration"]
    # I_s = define_stimulus(
    #     mesh=ep_mesh,
    #     chi=chi,
    #     C_m=C_m,
    #     time=time,
    #     stim_region=stim_region,
    #     stim_start=config["stim"]["start"],
    #     A=config["stim"]["amplitude"],
    #     duration=config["stim"]["duration"],
    # )

    S1_marker = 1
    S1_subdomain = dolfin.CompiledSubDomain(
        " ".join(
            (
                f"x[0] >= {stim_region[0][0]}",
                f"&& x[0] <= {stim_region[0][1]}",
                f"&& x[1] >= {stim_region[1][0]}",
                f"&& x[1] <= {stim_region[1][1]}",
                f"&& x[2] >= {stim_region[2][0]}",
                f"&& x[2] <= {stim_region[2][1]}",
            )
        )
    )

    S1_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)
    # with dolfin.XDMFFile((outdir / "stim_region_markers.xdmf").as_posix()) as xdmf:
    #     xdmf.write(S1_markers)

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms

    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=stim_start,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )

    dx = dolfin.Measure("dx", domain=mesh, subdomain_data=S1_markers)(S1_marker)
    return beat.base_model.Stimulus(dz=dx, expr=I_s)


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


def create_boundary_conditions(
    ffun_bcs,
    bcs_dirichlet,
    bcs_neumann,
):  # TODO: update to not need separate dirichlet and neumann list
    def dirichlet_bc(W):
        bcs_W = {
            "u_x": W.sub(0).sub(0),
            "u_y": W.sub(0).sub(1),
            "u_z": W.sub(0).sub(2),
            # TODO: add the rest (check dolfin doc)
        }

        bcs = []
        for bc in bcs_dirichlet:
            bcs.append(
                dolfin.DirichletBC(
                    bcs_W[bc["V"]],
                    bc["expression"],  # TODO: use dolfin expression
                    ffun_bcs,
                    bc["marker"],
                )
            )
        return bcs

    neumann_bc = []
    if bcs_neumann is not None:
        for bc in bcs_neumann:
            neumann_bc.append(
                pulse.NeumannBC(
                    traction=simcardems2.utils.float_to_constant(bc["expression"]),
                    marker=bc["marker"],
                )
            )

    # Collect Boundary Conditions
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=neumann_bc)
    return bcs


def handle_bcs(config, t_bcs):
    bcs_dirichlet = []
    bcs_neumann = []
    for bc in deepcopy(config["mechanics"]["bcs"]):
        if bc["type"] == "Dirichlet":
            bcs_dirichlet.append(bc)
        elif bc["type"] == "Neumann":
            bcs_neumann.append(bc)
        else:
            raise KeyError(
                f'{bc["type"]} is not a valid type of boundary '
                'condition. Use Dirichlet or Neumann. Check config file'
            )

        if bc["param_numbers"] > 0:
            bcs_parameters = {}
            for param_nr in range(bc["param_numbers"]):
                param_name = bc["param"][f"{param_nr}"]["name"]
                param_value = bc["param"][f"{param_nr}"]["value"]
                bcs_parameters[param_name] = param_value
            bc["expression"] = dolfin.Expression(
                bc["expression"],
                **bcs_parameters,
                t=t_bcs,
                degree=bc["degree"],
            )
        else:
            bc["expression"] = dolfin.Constant(0)
    return bcs_dirichlet, bcs_neumann


def read_mesh(config, refinement_level=0):
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(f'{config["sim"]["mech_mesh"]}.xdmf') as infile:
        infile.read(mesh)
    print(f'Loaded mesh: {config["sim"]["mech_mesh"]}')

    ep_mesh = mesh
    if refinement_level > 0:
        for i in range(refinement_level):
            ep_mesh = dolfin.adapt(ep_mesh)

    return mesh, ep_mesh


def compute_function_average_over_mesh(func, mesh):
    volume = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(domain=mesh))
    return dolfin.assemble(func * dolfin.dx(domain=mesh)) / volume


def setup_monodomain_model(config, mesh):
    sigma = [
        config["ep"]["conductivities"]["sigma_il"],
        config["ep"]["conductivities"]["sigma_it"],
        config["ep"]["conductivities"]["sigma_el"],
        config["ep"]["conductivities"]["sigma_et"],
    ]
    chi = config["ep"]["chi"]
    C_m = config["ep"]["C_m"]
    time = dolfin.Constant(0.0)
    I_s = define_stimulus(mesh, chi, C_m, time, config)
    M = define_conductivity_tensor(sigma, chi, C_m)
    params = {"preconditioner": "sor", "use_custom_preconditioner": False}
    return beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s, params=params)


def setup_mechancis_model(config, mesh, LandModel, land_params, t_bcs):
    ffun_bcs = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    marker_file = Path(f'{config["sim"]["markerfile"].rstrip(".xdmf")}.xdmf')

    with dolfin.XDMFFile(marker_file.as_posix()) as infile:
        infile.read(ffun_bcs)
    print(f"Loaded markerfile for bcs: {marker_file}")

    material_parameters = dict(
        a=config["mechanics"]["material"]["a"],
        a_f=config["mechanics"]["material"]["a_f"],
        b=config["mechanics"]["material"]["b"],
        b_f=config["mechanics"]["material"]["b_f"],
        a_s=config["mechanics"]["material"]["a_s"],
        b_s=config["mechanics"]["material"]["b_s"],
        a_fs=config["mechanics"]["material"]["a_fs"],
        b_fs=config["mechanics"]["material"]["b_fs"],
    )
    marker_functions = pulse.MarkerFunctions(ffun=ffun_bcs)

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
        mesh=mesh,
        eta=0,  # Fraction of transverse active tesion for active stress formulation.
        # 0 = active only along fiber, 1 = equal forces in all directions
        # (default=0.0).
        dLambda_tol=1e-12,
        **land_params,
    )
    active_model.t = dolfin.Constant(0.0)

    # Validate mechanics variables to output
    # for out_mech_var in list(set(out_mech_coord_names) | set(out_mech_var_names)):
    #     assert (
    #         out_mech_var in mech_variables
    #     ), f"Error: '{out_mech_var}' is not a valid variable name. Check config file"

    material = pulse.HolzapfelOgden(
        parameters=material_parameters,
        active_model=active_model,
        f0=f0,
        s0=s0,
        n0=n0,
    )

    bcs_dirichlet, bcs_neumann = handle_bcs(config, t_bcs)

    # Collect Boundary Conditions
    bcs = create_boundary_conditions(
        ffun_bcs,
        bcs_dirichlet,
        bcs_neumann,
    )

    problem = mechanicssolver.MechanicsProblem(geometry, material, bcs)
    problem.solve(0.0, 0.0)
    return problem


@dataclass
class DataCollector:
    problem: mechanicssolver.MechanicsProblem
    ep_ode_space: dolfin.FunctionSpace
    config: dict
    mech_variables: dict[str, dolfin.Function]

    def __post_init__(self):
        self.outdir.mkdir(exist_ok=True, parents=True)
        self._t = np.arange(0, self.config["sim"]["sim_dur"], self.config["sim"]["dt"])
        (self.outdir / "config.txt").write_text(toml.dumps(self.config))

        self.out_ep_var_names = self.config["output"]["all_ep"]
        self.out_mech_var_names = self.config["output"]["all_mech"]

        self.out_ep_coord_names = [f["name"] for f in self.config["output"]["point_ep"]]
        self.ep_coords = [
            [f[f"{coord}"] for coord in ["x", "y", "z"]] for f in self.config["output"]["point_ep"]
        ]

        self.out_mech_coord_names = [f["name"] for f in self.config["output"]["point_mech"]]
        self.mech_coords = [
            [f[f"{coord}"] for coord in ["x", "y", "z"]]
            for f in self.config["output"]["point_mech"]
        ]

        # Create function spaces for ep variables to output
        self.out_ep_funcs = {}
        for out_ep_var in self.out_ep_names:
            self.out_ep_funcs[out_ep_var] = dolfin.Function(self.ep_ode_space)

        self.out_ep_files = {}
        for out_ep_var in self.out_ep_var_names:
            self.out_ep_files[out_ep_var] = self.outdir / f"{out_ep_var}_out_ep.xdmf"
            self.out_ep_files[out_ep_var].unlink(missing_ok=True)
            self.out_ep_files[out_ep_var].with_suffix(".h5").unlink(missing_ok=True)

        self.out_mech_files = {}
        for out_mech_var in self.out_mech_var_names:
            self.out_mech_files[out_mech_var] = self.outdir / f"{out_mech_var}_out_mech.xdmf"
            self.out_mech_files[out_mech_var].unlink(missing_ok=True)
            self.out_mech_files[out_mech_var].with_suffix(".h5").unlink(missing_ok=True)

        self.out_ep_example_nodes = {}
        self.out_ep_volume_average_timeseries = {}
        for out_ep_var in self.out_ep_coord_names:
            self.out_ep_example_nodes[out_ep_var] = np.zeros(len(self.t))
            self.out_ep_volume_average_timeseries[out_ep_var] = np.zeros(len(self.t))

        self.out_mech_example_nodes = {}
        self.out_mech_volume_average_timeseries = {}
        for out_mech_var in self.out_mech_coord_names:
            self.out_mech_example_nodes[out_mech_var] = np.zeros(len(self.t))
            self.out_mech_volume_average_timeseries[out_mech_var] = np.zeros(len(self.t))

        self.timers = Timers()

    @property
    def t(self):
        return self._t

    @property
    def out_ep_names(self):
        return list(set(self.out_ep_var_names) | set(self.out_ep_coord_names))

    @property
    def outdir(self):
        return Path(self.config["sim"]["outdir"])

    @property
    def disp_file(self):
        return self.outdir / "displacement.xdmf"

    @property
    def ep_mesh(self):
        return self.ep_ode_space.mesh()

    def write_node_data_ep(self, i):
        # Store values to plot time series for given coord
        for var_nr, data in enumerate(self.config["output"]["point_ep"]):
            # Trace variable in coordinate
            out_ep_var = data["name"]
            self.out_ep_example_nodes[out_ep_var][i] = self.out_ep_funcs[out_ep_var](
                self.ep_coords[var_nr]
            )
            # Compute volume averages
            self.out_ep_volume_average_timeseries[out_ep_var][i] = (
                compute_function_average_over_mesh(self.out_ep_funcs[out_ep_var], self.ep_mesh)
            )

    def write_node_data_mech(self, i):
        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            # Trace variable in coordinate
            self.out_mech_example_nodes[out_mech_var][i] = self.mech_variables[out_mech_var](
                self.mech_coords[var_nr]
            )

            # Compute volume averages
            self.out_mech_volume_average_timeseries[out_mech_var][i] = (
                compute_function_average_over_mesh(
                    self.mech_variables[out_mech_var], self.problem.geometry.mesh
                )
            )

    def write_disp(self, j):
        U, p = self.problem.state.split(deepcopy=True)
        with dolfin.XDMFFile(self.disp_file.as_posix()) as file:
            file.write_checkpoint(U, "disp", j, dolfin.XDMFFile.Encoding.HDF5, True)

    def write_ep(self, j):
        for out_ep_var in self.out_ep_var_names:
            with dolfin.XDMFFile(self.out_ep_files[out_ep_var].as_posix()) as file:
                file.write_checkpoint(
                    self.out_ep_funcs[out_ep_var],
                    out_ep_var,
                    j,
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )
        for out_mech_var in self.out_mech_var_names:
            with dolfin.XDMFFile(self.out_mech_files[out_mech_var].as_posix()) as file:
                file.write_checkpoint(
                    self.mech_variables[out_mech_var],
                    out_mech_var,
                    j,
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )

    def finalize(self, inds, plot_results=True):
        self.timers.finalize(outdir=self.outdir)
        # Write averaged results for later analysis
        for out_ep_var in self.out_ep_coord_names:
            # with open(Path(outdir / f"{out_ep_var}_out_ep_volume_average.txt"), "w") as f:
            np.savetxt(
                self.outdir / f"{out_ep_var}_out_ep_volume_average.txt",
                self.out_ep_volume_average_timeseries[out_ep_var][inds],
            )

        for out_mech_var in self.out_mech_coord_names:
            # with open(Path(outdir / f"{out_mech_var}_out_mech_volume_average.txt"), "w") as f:
            np.savetxt(
                self.outdir / f"{out_mech_var}_out_mech_volume_average.txt",
                self.out_mech_volume_average_timeseries[out_mech_var][inds],
            )

        # Write point traces for later analysis
        for var_nr, data in enumerate(self.config["output"]["point_ep"]):
            out_ep_var = data["name"]
            path = (
                self.outdir
                / f"{out_ep_var}_ep_coord{self.ep_coords[var_nr][0]},{self.ep_coords[var_nr][1]},{self.ep_coords[var_nr][2]}.txt".replace(  # noqa: E501
                    " ", ""
                )
            )
            np.savetxt(path, self.out_ep_example_nodes[out_ep_var][inds])

        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            path = (
                self.outdir
                / f"{out_mech_var}_mech_coord{self.mech_coords[var_nr][0]},{self.mech_coords[var_nr][1]},{self.mech_coords[var_nr][2]}.txt"  # noqa: E501
            )
            np.savetxt(path, self.out_mech_example_nodes[out_mech_var][inds])

        print(f"Solved on {100 * len(inds) / len(self.t)}% of the time steps")
        inds = np.array(inds)

        if plot_results:
            fig, ax = plt.subplots(len(self.out_ep_coord_names), 1, figsize=(10, 10))
            if len(self.out_ep_coord_names) == 1:
                ax = np.array([ax])
            for i, out_ep_var in enumerate(self.out_ep_coord_names):
                ax[i].plot(self.t[inds], self.out_ep_volume_average_timeseries[out_ep_var][inds])
                ax[i].set_title(f"{out_ep_var} volume average")
                ax[i].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_ep_volume_averages.png")

            fig, ax = plt.subplots(len(self.out_ep_coord_names), 1, figsize=(10, 10))
            if len(self.out_ep_coord_names) == 1:
                ax = np.array([ax])
            for var_nr, data in enumerate(self.config["output"]["point_ep"]):
                out_ep_var = data["name"]
                ax[var_nr].plot(self.t[inds], self.out_ep_example_nodes[out_ep_var][inds])
                ax[var_nr].set_title(f"{out_ep_var} in coord {self.ep_coords[var_nr]}")
                ax[var_nr].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_ep_coord.png")

            fig, ax = plt.subplots(len(self.out_mech_coord_names), 1, figsize=(10, 10))
            if len(self.out_mech_coord_names) == 1:
                ax = np.array([ax])
            for i, out_mech_var in enumerate(self.out_mech_coord_names):
                ax[i].plot(
                    self.t[inds], self.out_mech_volume_average_timeseries[out_mech_var][inds]
                )
                ax[i].set_title(f"{out_mech_var} volume average")
                ax[i].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_mech_volume_averages.png")

            fig, ax = plt.subplots(len(self.out_mech_coord_names), 1, figsize=(10, 10))
            if len(self.out_mech_coord_names) == 1:
                ax = np.array([ax])

            for var_nr, data in enumerate(self.config["output"]["point_mech"]):
                out_mech_var = data["name"]
                ax[var_nr].plot(self.t[inds], self.out_mech_example_nodes[out_mech_var][inds])
                ax[var_nr].set_title(f"{out_mech_var} in coord {self.mech_coords[var_nr]}")
                ax[var_nr].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_mech_coord.png")


@dataclass
class Timers:
    timings_solveloop: list[float] = field(default_factory=list)
    timings_ep_steps: list[float] = field(default_factory=list)
    timings_mech_steps: list[float] = field(default_factory=list)
    no_of_newton_iterations: list[float] = field(default_factory=list)
    timings_var_transfer: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.start_total()

    def start_total(self):
        self.total_timer = dolfin.Timer("total")

    def stop_total(self):
        self.total_timer.stop()
        self.timings_total = self.total_timer.elapsed

    def start_ep(self):
        self.ep_timer = dolfin.Timer("ep")

    def stop_ep(self):
        self.ep_timer.stop()
        self.timings_ep_steps.append(self.ep_timer.elapsed()[0])

    def start_single_loop(self):
        self.timing_single_loop = dolfin.Timer("single_loop")

    def stop_single_loop(self):
        self.timing_single_loop.stop()
        self.timings_solveloop.append(self.timing_single_loop.elapsed()[0])

    def start_var_transfer(self):
        self.timing_var_transfer = dolfin.Timer("mv and lambda transfer time")

    def stop_var_transfer(self):
        self.timing_var_transfer.stop()

    def collect_var_transfer(self):
        self.timings_var_transfer.append(self.timing_var_transfer.elapsed()[0])

    def start_mech(self):
        self.mech_timer = dolfin.Timer("mech time")

    def stop_mech(self):
        self.mech_timer.stop()
        self.timings_mech_steps.append(self.mech_timer.elapsed()[0])

    def finalize(self, outdir: Path):
        self.stop_total()
        timings = dolfin.timings(
            dolfin.TimingClear.keep,
            [dolfin.TimingType.wall, dolfin.TimingType.user, dolfin.TimingType.system],
        ).str(True)
        print(timings)
        # with open(Path(outdir / "solve_timings.txt"), "w") as f:
        #     f.write("Loop total times\n")
        #     np.savetxt(f, self.timings_solveloop)
        #     f.write("Ep steps times\n")
        #     np.savetxt(f, self.timings_ep_steps)
        #     f.write("Mech steps times\n")
        #     np.savetxt(f, self.timings_mech_steps)
        #     f.write("No of mech iterations\n")
        #     np.savetxt(f, self.no_of_newton_iterations, fmt="%s")
        #     f.write("mv and lambda transfer time\n")
        #     np.savetxt(f, self.timings_var_transfer)
        #     f.write("Total time\n")
        #     f.write(f"{self.total_timer.elapsed()[0]}\n")
        #     f.write(timings)

        (outdir / "solve_timings.json").write_text(
            json.dumps(
                {
                    "Loop total times": self.timings_solveloop,
                    "Ep steps times": self.timings_ep_steps,
                    "Mech steps times": self.timings_mech_steps,
                    "No of mech iterations": self.no_of_newton_iterations,
                    "mv and lambda transfer time": self.timings_var_transfer,
                    "Total time": self.total_timer.elapsed()[0],
                    "timings": timings,
                },
                indent=4,
            )
        )
