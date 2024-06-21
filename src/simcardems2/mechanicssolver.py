import dolfin
import pulse
import logging
import ufl_legacy as ufl
from collections import deque

logger = logging.getLogger(__name__)


def enlist(obj):
    try:
        return list(obj)
    except TypeError:
        return [obj]


class NonlinearProblem(dolfin.NonlinearProblem):
    def __init__(
        self,
        J,
        F,
        bcs,
        output_matrix=False,
        output_matrix_path="output",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._J = J
        self._F = F

        self.bcs = enlist(bcs)
        self.output_matrix = output_matrix
        self.output_matrix_path = output_matrix_path
        self.verbose = True
        self.n = 0

    def F(self, b: dolfin.PETScVector, x: dolfin.PETScVector):
        dolfin.assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A: dolfin.PETScMatrix, x: dolfin.PETScVector):
        dolfin.assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class ContinuationProblem(NonlinearProblem):
    def __init__(self, J, F, bcs, **kwargs):
        # self.problem = problem
        super().__init__(J, F, bcs, **kwargs)
        self._J = J
        self._F = F

        self.bcs = enlist(bcs)

        # super(ContinuationProblem, self).__init__()

        self.fres = deque(maxlen=2)

        self.first_call = True
        self.skipF = False

        self._assemble_jacobian = True

    def form(self, A, P, b, x):
        # pb = self.problem
        if self._assemble_jacobian:
            dolfin.assemble_system(self._J, self._F, self.bcs, A_tensor=A, b_tensor=b)
        else:
            dolfin.assemble(self._F, tensor=b)
            if self.bcs:
                for bc in self.bcs:
                    bc.apply(b)
        self._assemble_jacobian = not self._assemble_jacobian

        return
        # Timer("ContinuationSolver: form")
        # pb = self.problem

        # # check if we need to assemble the jacobian
        # if self.first_call:
        #     reset_jacobian = True
        #     self.first_call = False
        #     self.skipF = True
        # else:
        #     reset_jacobian = b.empty() and not A.empty()
        #     self.skipF = reset_jacobian

        #     if len(self.fres) == 2 and reset_jacobian:
        #         if self.fres[1] < 0.1 * self.fres[0]:
        #             debug("REUSE J")
        #             reset_jacobian = False

        # if reset_jacobian:
        #     # requested J, assemble both
        #     debug("ASSEMBLE J")
        #     assemble_system(pb.dG, pb.G, pb.bcs, x0=x, A_tensor=A, b_tensor=b)

    def J(self, A, x):
        pass

    def F(self, b, x):
        return
        # if self.skipF:
        #     return
        # pb = self.problem
        # assemble(pb.G, tensor=b)
        # for bc in pb.bcs:
        #     bc.apply(b)
        # self.fres.append(b.norm("l2"))


class NewtonSolver(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: pulse.NonlinearProblem,
        state: dolfin.Function,
        active,
        update_cb=None,
        parameters=None,
    ):
        self.active = active
        print(f"Initialize NewtonSolver with parameters: {parameters!r}")
        dolfin.PETScOptions.clear()
        self.dx = dolfin.Measure("dx", domain=state.function_space().mesh())
        self.volume = dolfin.assemble(dolfin.Constant(1) * self.dx)
        self._problem = problem
        self._state = state
        self._update_cb = update_cb
        self._prev_state = dolfin.Vector(state.vector().copy())
        self._diff = dolfin.Vector(state.vector().copy())
        #
        # Initializing Newton solver (parent class)
        self.petsc_solver = dolfin.PETScKrylovSolver()
        super().__init__(
            self._state.function_space().mesh().mpi_comm(),
            self.petsc_solver,
            dolfin.PETScFactory.instance(),
        )
        self._handle_parameters(parameters)
        # self.dt_mech = 10.0
        # self.t_mech = 0.0
        # self.parameters["maximum_iterations"] = 50

    def _handle_parameters(self, parameters):
        # Setting default parameters
        params = type(self).default_solver_parameters()

        if parameters is not None:
            params.update(parameters)

        for k, v in params.items():
            if self.parameters.has_parameter(k):
                self.parameters[k] = v
            if self.parameters.has_parameter_set(k):
                for subk, subv in params[k].items():
                    self.parameters[k][subk] = subv
        petsc = params.pop("petsc", {})
        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.newton_verbose = params.pop("newton_verbose", False)
        self.ksp_verbose = params.pop("ksp_verbose", False)
        self.debug = params.pop("debug", False)
        if self.newton_verbose:
            dolfin.set_log_level(dolfin.LogLevel.INFO)
            self.parameters["report"] = True
        if self.ksp_verbose:
            self.parameters["lu_solver"]["report"] = True
            self.parameters["lu_solver"]["verbose"] = True
            self.parameters["krylov_solver"]["monitor_convergence"] = True
            dolfin.PETScOptions.set("ksp_monitor_true_residual")
        self.linear_solver().set_from_options()
        self._residual_index = 0
        self._residuals = []
        self.parameters["convergence_criterion"] = "incremental"
        self.parameters["relaxation_parameter"] = 0.8

    # def register_datacollector(self, datacollector):
    #     self._datacollector = datacollector

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "preonly",
                # "ksp_type": "gmres",
                # "pc_type": "lu",
                "pc_type": "cholesky",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
                "mat_mumps_icntl_7": 6,
            },
            "newton_verbose": False,
            "ksp_verbose": False,
            "debug": False,
            "linear_solver": "gmres",
            # "preconditioner": "lu",
            # "linear_solver": "mumps",
            "error_on_nonconvergence": False,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 100,
            "report": False,
            "krylov_solver": {
                "nonzero_initial_guess": False,
                "absolute_tolerance": 1e-13,
                "relative_tolerance": 1e-13,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": True, "verbose": False},
        }

    def converged(self, r, p, i):
        self._converged_called = True

        # breakpoint()
        res = r.norm("l2")
        print(f"Mechanics solver residual: {res}")
        # if res > 0.1:
        #     breakpoint()
        # if i > 17:
        #     breakpoint()

        if self.debug:
            if not hasattr(self, "_datacollector"):
                print("No datacollector registered with the NewtonSolver")

            else:
                self._residuals.append(res)

        return super().converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

    def super_solve(self):
        return super().solve(self._problem, self._state.vector())

    def solve(self, t0: float, dt: float) -> tuple[int, bool]:
        self.t0 = t0
        self.dt = dt

        self.active._t_prev = t0

        self.active.t = t0 + dt
        print("Solving mechanics")

        self._solve_called = True

        nit, conv = self.super_solve()

        print("After: ", self._state.vector().get_local()[:10])

        if not conv:
            raise RuntimeError("Newton solver did not converge")

        self._diff.zero()

        self._diff.axpy(1.0, self._state.vector())
        self._diff.axpy(-1.0, self._prev_state)
        print(f"Difference between previous state : {self._diff.norm('l2')}")

        self._prev_state.zero()
        self._prev_state.axpy(1.0, self._state.vector())
        print("Done solving mechanics")

        return (nit, conv)

    def reset(self):
        self._state.vector().zero()
        self._state.vector().axpy(1.0, self._prev_state)
        self.ode.reset()

    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True

        # Update x from the dx obtained from linear solver (Newton iteration) :
        # x = -rp*dx (rp : relax param)
        print(f"Updating mechanics solution with relax parameter {rp}, iteration {i}")

        super().update_solution(x, dx, rp, p, i)

        if self._update_cb is not None:
            self._update_cb(x)


class GuccioneMaterial:
    def __init__(self, params):
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters():
        p = {
            "C": 2.0,
            "bf": 8.0,
            "bt": 2.0,
            "bfs": 4.0,
            "e1": None,
            "e2": None,
            "e3": None,
            "kappa": None,
            "Tactive": None,
        }
        return p

    def is_isotropic(self):
        """
        Return True if the material is isotropic.
        """
        p = self._parameters
        return p["bt"] == 1.0 and p["bf"] == 1.0 and p["bfs"] == 1.0

    def is_incompressible(self):
        """
        Return True if the material is incompressible.
        """
        return self._parameters["kappa"] is None

    def strain_energy(self, F, p=None):
        """
        UFL form of the strain energy.
        """
        params = self._parameters

        I = ufl.Identity(3)
        J = ufl.det(F)
        C = pow(J, -float(2) / 3) * F.T * F
        E = 0.5 * (C - I)

        CC = dolfin.Constant(params["C"], name="C")
        if self.is_isotropic():
            # isotropic case
            Q = ufl.inner(E, E)
        else:
            # fully anisotropic
            bt = dolfin.Constant(params["bt"], name="bt")
            bf = dolfin.Constant(params["bf"], name="bf")
            bfs = dolfin.Constant(params["bfs"], name="bfs")

            e1 = params["e1"]
            e2 = params["e2"]
            e3 = params["e3"]

            E11, E12, E13 = (
                ufl.inner(E * e1, e1),
                ufl.inner(E * e1, e2),
                ufl.inner(E * e1, e3),
            )
            E21, E22, E23 = (
                ufl.inner(E * e2, e1),
                ufl.inner(E * e2, e2),
                ufl.inner(E * e2, e3),
            )
            E31, E32, E33 = (
                ufl.inner(E * e3, e1),
                ufl.inner(E * e3, e2),
                ufl.inner(E * e3, e3),
            )

            Q = (
                bf * E11**2
                + bt * (E22**2 + E33**2 + E23**2 + E32**2)
                + bfs * (E12**2 + E21**2 + E13**2 + E31**2)
            )

        # passive strain energy
        Wpassive = CC / 2.0 * (ufl.exp(Q) - 1)

        # active strain energy
        if params["Tactive"] is not None:
            self.Tactive = params["Tactive"]
            I4 = ufl.inner(C * e1, e1)
            Wactive = self.Tactive / 2.0 * (I4 - 1)
        else:
            Wactive = 0.0

        # incompressibility
        if params["kappa"] is not None:
            kappa = dolfin.Constant(params["kappa"], name="kappa")
            Winc = kappa * (J**2 - 1 - 2 * ufl.ln(J))
        else:
            Winc = -p * (J - 1)

        return Wpassive + Wactive + Winc


class MechanicsProblem(pulse.MechanicsProblem):
    def _init_solver(self):
        if hasattr(self, "_dirichlet_bc"):
            bcs = self._dirichlet_bc
        else:
            bcs = []

        self._problem = NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=bcs,
        )
        self._problem = ContinuationProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=bcs,
        )

        self.solver = NewtonSolver(
            problem=self._problem,
            state=self.state,
            active=self.material.active,
        )

    def solve(self, t0: float, dt: float):
        self._init_forms()
        return self.solver.solve(t0, dt)
