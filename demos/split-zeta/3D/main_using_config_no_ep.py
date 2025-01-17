import logging
from pathlib import Path
import numpy as np
import dolfin
import pulse
import matplotlib.pyplot as plt


from simcardems2 import mechanicssolver
from simcardems2.land_Zetasplit import LandModel


try:
    raise ImportError
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


logging.getLogger("beat").setLevel(logging.ERROR)

mesh = dolfin.Mesh()
with dolfin.XDMFFile("mesh_mech_0.5dx_0.5Lx_1Ly_2Lz.xdmf") as infile:
    infile.read(mesh)


ffun_bcs = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with dolfin.XDMFFile("mesh_mech_0.5dx_0.5Lx_1Ly_2Lz_surface_ffun.xdmf") as infile:
    infile.read(ffun_bcs)


material_parameters = dict(
    a=2.28,
    a_f=1.685,
    b=9.726,
    b_f=15.779,
    a_s=0.0,
    b_s=0.0,
    a_fs=0.0,
    b_fs=0.0,
)
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4


time = dolfin.Constant(0.0)

mechanics_missing_values_ = np.zeros(2)

# Set the activation
activation_space = dolfin.FunctionSpace(mesh, "DG", 1)
activation = dolfin.Function(activation_space)
num_points_mech = activation.vector().local_size()
marker_functions = pulse.MarkerFunctions(ffun=ffun_bcs)


def create_boundary_conditions(
    ffun_bcs,
):  # TODO: update to not need separate dirichlet and neumann list
    def dirichlet_bc(W):
        bcs_W = {
            "u_x": W.sub(0).sub(0),
            "u_y": W.sub(0).sub(1),
            "u_z": W.sub(0).sub(2),
            # TODO: add the rest (check dolfin doc)
        }

        bcs = []
        for V, marker in [("u_x", 1), ("u_y", 3), ("u_z", 5)]:
            bcs.append(
                dolfin.DirichletBC(
                    bcs_W[V],
                    0.0,
                    ffun_bcs,
                    marker,
                )
            )
        return bcs

    # Collect Boundary Conditions
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    return bcs


f0 = dolfin.as_vector([1.0, 0.0, 0.0])
s0 = dolfin.as_vector([0.0, 1.0, 0.0])
n0 = dolfin.as_vector([0.0, 0.0, 1.0])
microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)
geometry = pulse.Geometry(
    mesh=mesh, marker_functions=marker_functions, microstructure=microstructure
)

XS = dolfin.Function(activation_space)
XW = dolfin.Function(activation_space)

active_model = LandModel(
    f0=f0,
    s0=s0,
    n0=n0,
    XS=XS,
    XW=XW,
    mesh=mesh,
    eta=0,
    dLambda_tol=1e-12,
)
active_model.t = 0.0

sigma_ff = dolfin.Function(activation_space)
sigma_ff_active = dolfin.Function(activation_space)
sigma_ff_passive = dolfin.Function(activation_space)


mech_variables = {
    "Ta": active_model.Ta_current,
    "Zetas": active_model._Zetas,
    "Zetaw": active_model._Zetaw,
    "lambda": active_model.lmbda,
    "XS": active_model.XS,
    "XW": active_model.XW,
    "dLambda": active_model._dLambda,
    "sigma_ff": sigma_ff,
    "sigma_ff_active": sigma_ff_active,
    "sigma_ff_passive": sigma_ff_passive,
}

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
bcs = create_boundary_conditions(ffun_bcs)

problem = mechanicssolver.MechanicsProblem(geometry, material, bcs)
problem.solve(0.0, 0.0)


N = 400
dt_ep = 0.05

dt = 0.05 * N

inds = []  # Array with time-steps for which we solve mechanics
j = 0

outdir = Path("no-ep")
outdir.mkdir(exist_ok=True)
disp_file = outdir / "disp.xdmf"
disp_file.unlink(missing_ok=True)
disp_file.with_suffix(".h5").unlink(missing_ok=True)

lmbda_saved = dolfin.Function(activation_space)

timer = dolfin.Timer("solve_loop")
Tas = []
Tas_saved = []
lmbdas = []
lmbdas_saved = []
for i in range(20):
    ti = i * dt
    print(f"Solving time {ti:.2f} ms")

    with dolfin.XDMFFile("100ms_N10_zeta_split/XW_out_mech.xdmf") as infile:
        infile.read_checkpoint(XW, "XW", i)

    with dolfin.XDMFFile("100ms_N10_zeta_split/XS_out_mech.xdmf") as infile:
        infile.read_checkpoint(XS, "XS", i)

    with dolfin.XDMFFile("100ms_N10_zeta_split/Zetas_out_mech.xdmf") as infile:
        infile.read_checkpoint(active_model._Zetas, "Zetas", i)

    with dolfin.XDMFFile("100ms_N10_zeta_split/Zetaw_out_mech.xdmf") as infile:
        infile.read_checkpoint(active_model._Zetaw, "Zetaw", i)

    with dolfin.XDMFFile("100ms_N10_zeta_split/Ta_out_mech.xdmf") as infile:
        infile.read_checkpoint(active_model.Ta_current, "Ta", i)

    with dolfin.XDMFFile("100ms_N10_zeta_split/lambda_out_mech.xdmf") as infile:
        infile.read_checkpoint(lmbda_saved, "lambda", i)

    print("Solve mechanics")
    # active_model._t_prev = ti

    active_model.t = ti + dt
    problem.solve(ti, dt)
    active_model.update_prev()

    U, p = problem.state.split(deepcopy=True)

    Ta = active_model.Ta(active_model.lmbda)
    Ta_mean = dolfin.assemble(Ta * dolfin.dx)
    lmbda_mean = dolfin.assemble(active_model.lmbda * dolfin.dx)
    lmbdas.append(lmbda_mean)
    lmbdas_saved.append(dolfin.assemble(lmbda_saved * dolfin.dx))
    Tas.append(Ta_mean)
    Tas_saved.append(dolfin.assemble(active_model.Ta_current * dolfin.dx))

    with dolfin.XDMFFile(disp_file.as_posix()) as file:
        file.write_checkpoint(U, "disp", j, dolfin.XDMFFile.Encoding.HDF5, True)


print(np.argmax(Tas))
print(np.argmin(lmbdas))
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(Tas, label="Computed")
ax[0].plot(Tas_saved, linestyle="--", label="Saved")
# ax[0].plot(active_model.Ta_before, label="Ta before")
# ax[0].plot(active_model.Ta_after, linestyle="--", label="Ta after")
ax[0].set_ylabel("Ta")
ax[1].plot(lmbdas, label="Computed")
ax[1].plot(lmbdas_saved, linestyle="--", label="Saved")
# ax[1].plot(active_model.lmbda_before, label="Lambda before")
# ax[1].plot(active_model.lmbda_after, linestyle="--", label="Lambda after")
ax[1].set_ylabel("Lambda")
for axi in ax:
    axi.grid()
    axi.legend()
fig.savefig(outdir / "Ta.png")
# breakpoint()
