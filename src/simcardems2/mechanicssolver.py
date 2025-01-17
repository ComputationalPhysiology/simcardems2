import dolfin
import pulse
import logging
import ufl_legacy as ufl
from typing import Tuple

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
        self._assemble_jacobian = True

    def form(self, A, P, b, x):
        if self._assemble_jacobian:
            dolfin.assemble_system(self._J, self._F, self.bcs, A_tensor=A, b_tensor=b)
        else:
            dolfin.assemble(self._F, tensor=b)
            if self.bcs:
                for bc in self.bcs:
                    bc.apply(b)
        self._assemble_jacobian = not self._assemble_jacobian

    def J(self, A, x):
        pass

    def F(self, b, x):
        return


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
        logger.info(f"Initialize NewtonSolver with parameters: {parameters!r}")
        dolfin.PETScOptions.clear()
        self.dx = dolfin.Measure("dx", domain=state.function_space().mesh())
        self.volume = dolfin.assemble(dolfin.Constant(1) * self.dx)
        self._problem = problem
        self._state = state
        self._update_cb = update_cb
        self._tmp_state = dolfin.Function(state.function_space())
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
        logger.info(f"Mechanics solver residual: {res}")

        if self.debug:
            self._residuals.append(res)

        return super().converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

    def super_solve(self):
        return super().solve(self._problem, self._state.vector())

    def solve(self, t0: float, dt: float) -> Tuple[int, bool]:
        self.t0 = t0
        self.dt = dt

        logger.info("Solving mechanics")

        self._solve_called = True

        nit, conv = self.super_solve()

        u, p = self._state.split(deepcopy=True)

        F = ufl.grad(u) + ufl.Identity(3)
        f = F * self.active.f0
        lmbda = dolfin.sqrt(f**2)

        self.active._projector.project(self.active.lmbda, lmbda)
        if self.active.dt > 0:
            self.active._projector.project(
                self.active._dLambda, (lmbda - self.active.lmbda_prev) / self.active.dt
            )

        self.active._projector.project(self.active.Ta_current, self.active.Ta(lmbda))
        self.active.update_current(lmbda=lmbda)
        self.active.update_prev()
        if not conv:
            raise RuntimeError("Newton solver did not converge")

        self._diff.zero()

        self._diff.axpy(1.0, self._state.vector())
        self._diff.axpy(-1.0, self._prev_state)
        self._prev_state.zero()
        self._prev_state.axpy(1.0, self._state.vector())

        return (nit, conv)

    def reset(self):
        self._state.vector().zero()
        self._state.vector().axpy(1.0, self._prev_state)
        self.ode.reset()

    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True

        # Update x from the dx obtained from linear solver (Newton iteration) :
        # x = -rp*dx (rp : relax param)
        logger.debug(f"Updating mechanics solution with relax parameter {rp}, iteration {i}")

        super().update_solution(x, dx, rp, p, i)

        if self._update_cb is not None:
            self._update_cb(x)


class MechanicsProblem(pulse.MechanicsProblem):
    def _init_forms(self):
        logger.debug("Initialize forms mechanics problem")
        # Displacement and hydrostatic_pressure
        u, p = dolfin.split(self.state)
        v, q = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(ufl.grad(u) + ufl.Identity(3))
        J = ufl.det(F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            F,
        ) + self.material.compressibility(p, J)

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )
        f0 = self.material.active.f0
        f = F * f0
        lmbda = ufl.sqrt(f**2)
        Pa = self.material.active.Ta(lmbda) * ufl.outer(f, f0)
        self._virtual_work += ufl.inner(Pa, ufl.grad(v)) * dx

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()

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

        self.solver = NewtonSolver(
            problem=self._problem,
            state=self.state,
            active=self.material.active,
        )

    def solve(self, t0: float, dt: float):
        self._init_forms()
        return self.solver.solve(t0, dt)
