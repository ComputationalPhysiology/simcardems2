import dolfin
import ufl_legacy as ufl
import pulse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# def load_timesteps_from_xdmf(xdmffile):
#     times = {}
#     i = 0
#     tree = ET.parse(xdmffile)
#     for elem in tree.iter():
#         if elem.tag == "Time":
#             times[i] = float(elem.get("Value"))
#             i += 1

#     return times


mesh = dolfin.Mesh()
with dolfin.XDMFFile(f'mesh_mech_0.5dx_0.5Lx_1Ly_2Lz.xdmf') as infile:
    infile.read(mesh)



ffun_bcs = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with dolfin.XDMFFile(f'mesh_mech_0.5dx_0.5Lx_1Ly_2Lz_surface_ffun.xdmf') as infile:
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

fig_mean, ax_mean = plt.subplots(2, 1, sharex=True)
fig_u, ax_u = plt.subplots(3, 2, sharex=True, sharey="row")
for k, N in enumerate([1, 400]):
    xdmffile_ta = f"check_Ta/N{N}/Ta_out_mech.xdmf"
    xdmffile_u = f"check_Ta/N{N}/disp.xdmf"

    if N == 1:
        times = [400 * i for i in range(25)]
    elif N == 400:
        times = list(range(25))

    timepoint = 50
    W = dolfin.FunctionSpace(mesh, "DG", 1)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    U = dolfin.Function(V)
    Ta_mean = []
    lmbda_mean = []
    lmbda_U_mean = []
    Tas = []
    lmbdas = []
    lmbdas_U = []
    Ta = dolfin.Function(W)
    lmbda = dolfin.Function(W)
    lmbda_U = dolfin.Function(W)

    ux = []
    uy = []
    uz = []
    Ux = []
    Uy = []
    Uz = []

    material = pulse.HolzapfelOgden(
        parameters=material_parameters,
        active_model="active_stress",
        activation=Ta,
        f0=f0,
        s0=s0,
        n0=n0,
    )

    # Collect Boundary Conditions
    bcs = create_boundary_conditions(ffun_bcs)

    problem = pulse.MechanicsProblem(geometry, material, bcs)

    problem.solve()

    for i in times:
        with dolfin.XDMFFile(xdmffile_ta) as infile:
            infile.read_checkpoint(Ta, "Ta", i)

        with dolfin.XDMFFile(xdmffile_u) as infile:
            infile.read_checkpoint(U, "disp", i)

        problem.solve()
        u, p = problem.state.split(deepcopy=True)
        F = ufl.grad(u) + ufl.Identity(3)
        lmda_expr = ufl.sqrt(ufl.inner(F * f0, F * f0))
        lmbda.assign(dolfin.project(lmda_expr, W))

        F_U = ufl.grad(U) + ufl.Identity(3)
        lmda_expr_U = ufl.sqrt(ufl.inner(F_U * f0, F_U * f0))
        lmbda_U.assign(dolfin.project(lmda_expr_U, W))

        ux.append(dolfin.assemble(u.sub(0) * dolfin.dx))
        uy.append(dolfin.assemble(u.sub(1) * dolfin.dx))
        uz.append(dolfin.assemble(u.sub(2) * dolfin.dx))
        Ux.append(dolfin.assemble(U.sub(0) * dolfin.dx))
        Uy.append(dolfin.assemble(U.sub(1) * dolfin.dx))
        Uz.append(dolfin.assemble(U.sub(2) * dolfin.dx))

        # ux.append(u.vector().get_local()[0::3])
        # uy.append(u.vector().get_local()[1::3])
        # uz.append(u.vector().get_local()[2::3])
        # Ux.append(U.vector().get_local()[0::3])
        # Uy.append(U.vector().get_local()[1::3])
        # Uz.append(U.vector().get_local()[2::3])



        Tas.append(Ta.vector().get_local())
        lmbdas.append(lmbda.vector().get_local())
        lmbdas_U.append(lmbda_U.vector().get_local())
        Ta_mean.append(dolfin.assemble(Ta * dolfin.dx))
        lmbda_mean.append(dolfin.assemble(lmda_expr * dolfin.dx))
        lmbda_U_mean.append(dolfin.assemble(lmda_expr_U * dolfin.dx))

    Tas = np.array(Tas)
    lmbdas = np.array(lmbdas)
    lmbdas_U = np.array(lmbdas_U)
    # ux = np.array(ux)
    # uy = np.array(uy)
    # uz = np.array(uz)

    ax_u[0, k].plot(ux, label="computed")
    ax_u[0, k].plot(Ux, label="saved")
    ax_u[0, k].set_title(N)
    ax_u[1, k].plot(uy, label="computed")
    ax_u[1, k].plot(Uy, label="saved")
    ax_u[2, k].plot(uz, label="computed")
    ax_u[2, k].plot(Uz, label="saved")






    fig, ax = plt.subplots(2, 2, sharex=True)
    ax[0, 0].plot(Tas[:, ::4])
    ax[0, 0].set_title("Ta")
    ax[0, 1].plot(lmbdas[:, ::4])
    ax[0, 1].set_title("lmbda")
    ax[1, 0].plot(Ta_mean)
    ax[1, 0].set_title("Ta mean")
    ax[1, 1].plot(lmbda_mean, label="lmbda")
    ax[1, 1].plot(lmbda_U_mean, label="lmbda (saved)")
    ax[1, 1].legend()
    ax[1, 1].set_title("lmbda mean")

    for axi in ax.flatten():
        axi.grid()
    fig.savefig(f"check_Ta/N{N}/Ta_points.png")


    ax_mean[0].plot(Ta_mean, label=f"N={N}")
    ax_mean[1].plot(lmbda_mean, label=f"N={N}")
    ax_mean[1].plot(lmbda_U_mean, label=f"N={N} (saved)")

for axi in ax_mean:
    axi.grid()
ax_mean[0].set_title("Ta mean")
ax_mean[1].set_title("lmbda mean")
ax_mean[0].legend()
ax_mean[1].legend()
fig_mean.savefig("check_Ta/mean.png")

ax_u[0, 0].set_ylabel("ux")
ax_u[1, 0].set_ylabel("uy")
ax_u[2, 0].set_ylabel("uz")
for axi in ax_u.flatten():
    axi.grid()
ax_u[1, 1].legend()
fig_u.savefig("check_Ta/u.png")

exit()
