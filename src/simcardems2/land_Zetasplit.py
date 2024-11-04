import pulse
import dolfin
import ufl_legacy as ufl
import logging
import numpy as np
from enum import Enum

from . import utils

logger = logging.getLogger(__name__)


class Scheme(str, Enum):
    fd = "fd"
    bd = "bd"
    analytic = "analytic"


def _Zeta(Zeta_prev, A, c, dLambda, dt, scheme: Scheme):
    # if scheme == Scheme.analytic:
    dZetas_dt = A * dLambda - Zeta_prev * c
    dZetas_dt_linearized = -c
    if abs(c) > 1e-8:
        return Zeta_prev + dZetas_dt * (np.exp(-c * dt) - 1.0) / dZetas_dt_linearized
    else:
        # Forward euler
        return Zeta_prev + dZetas_dt * dt


_parameters = {
    "Beta0": 2.3,
    "Tot_A": 25.0,
    "Tref": 120,
    "kuw": 0.182,
    "kws": 0.012,
    "phi": 2.23,
    "rs": 0.25,
    "rw": 0.5,
}


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        f0,
        s0,
        n0,
        XS,
        XW,
        mesh,
        parameters=None,
        Zetas=None,
        Zetaw=None,
        lmbda=None,
        eta=0,
        scheme: Scheme = Scheme.analytic,
        dLambda_tol: float = 1e-12,
        **kwargs,
    ):
        logger.debug("Initialize Land Model")
        super().__init__(f0=f0, s0=s0, n0=n0)

        self._eta = eta
        # self.function_space = dolfin.FunctionSpace(mesh, "DG", 0)
        self.function_space = dolfin.FunctionSpace(mesh, "DG", 1)
        # self.quad_space = pulse.QuadratureSpace(mesh, 4)
        self.u_space = dolfin.VectorFunctionSpace(mesh, "CG", 2)
        self.u = dolfin.Function(self.u_space)
        self.u_prev = dolfin.Function(self.u_space)

        self.XS = XS
        self.XW = XW
        if parameters is None:
            parameters = _parameters
        self._parameters = parameters

        self._scheme = scheme
        # self.Ta_before = []
        # self.Ta_after = []
        # self.lmbda_before = []
        # self.lmbda_after = []
        # self.times = []

        self._dLambda = dolfin.Function(self.function_space)
        self.lmbda_prev = dolfin.Function(self.function_space)
        self.lmbda_prev.vector()[:] = 1.0
        if lmbda is not None:
            self.lmbda_prev.assign(lmbda)
        self.lmbda = dolfin.Function(self.function_space)

        self._Zetas = dolfin.Function(self.function_space)
        self.Zetas_prev = dolfin.Function(self.function_space)
        if Zetas is not None:
            self.Zetas_prev.assign(Zetas)

        self._Zetaw = dolfin.Function(self.function_space)
        self.Zetaw_prev = dolfin.Function(self.function_space)
        if Zetaw is not None:
            self.Zetaw_prev.assign(Zetaw)

        self.Ta_current = dolfin.Function(self.function_space, name="Ta")
        self._projector = utils.Projector(self.function_space)
        self._dLambda_tol = dLambda_tol
        self._t_prev = 0.0

    # @property
    # def lmbda_prev(self):
    #     F = dolfin.grad(self.u_prev) + ufl.Identity(3)
    #     f = F * self.f0
    #     return dolfin.sqrt(f**2)

    # @property
    # def lmbda_u(self):
    #     F = dolfin.grad(self.u) + ufl.Identity(3)
    #     f = F * self.f0
    #     return dolfin.sqrt(f**2)

    # @property
    def dLambda(self, lmbda):
        logger.debug("Evaluate dLambda")
        if self.dt == 0:
            return self._dLambda
        else:
            return (lmbda - self.lmbda_prev) / self.dt

    @property
    def Aw(self):
        Tot_A = self._parameters["Tot_A"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        return (
            Tot_A
            * rs
            * scale_popu_rs
            / (rs * scale_popu_rs + rw * scale_popu_rw * (1.0 - (rs * scale_popu_rs)))
        )

    @property
    def As(self):
        return self.Aw

    @property
    def cw(self):
        phi = self._parameters["phi"]
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]

        scale_popu_kuw = 1.0  # self._parameters["scale_popu_kuw"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        return kuw * scale_popu_kuw * phi * (1.0 - (rw * scale_popu_rw)) / (rw * scale_popu_rw)

    @property
    def cs(self):
        phi = self._parameters["phi"]
        kws = self._parameters["kws"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_kws = 1.0  # self._parameters["scale_popu_kws"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        return (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1.0 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

    def update_Zetas(self, lmbda):
        logger.debug("update Zetas")
        self._projector(
            self._Zetas,
            _Zeta(
                self.Zetas_prev,
                self.As,
                self.cs,
                self.dLambda(lmbda),
                self.dt,
                self._scheme,
            ),
        )

    def Zetas(self, lmbda):
        # return self._Zetas
        return _Zeta(
            self.Zetas_prev,
            self.As,
            self.cs,
            self.dLambda(lmbda),
            self.dt,
            self._scheme,
        )

    def update_Zetaw(self, lmbda):
        logger.debug("update Zetaw")
        self._projector(
            self._Zetaw,
            _Zeta(
                self.Zetaw_prev,
                self.Aw,
                self.cw,
                self.dLambda(lmbda),
                self.dt,
                self._scheme,
            ),
        )

    def Zetaw(self, lmbda):
        return _Zeta(
            self.Zetaw_prev,
            self.Aw,
            self.cw,
            self.dLambda(lmbda),
            self.dt,
            self._scheme,
        )

    @property
    def dt(self) -> float:
        return self.t - self._t_prev

    def update_prev(self):
        logger.debug("update previous")
        self.Zetas_prev.vector()[:] = self._Zetas.vector()
        self.Zetaw_prev.vector()[:] = self._Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda.vector()
        # self.u_prev_prev.vector()[:] = self.u_prev.vector()
        self._t_prev = self.t

    def Ta(self, lmbda):
        logger.debug("Evaluate Ta")
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = 1.0  # self._parameters["scale_popu_Tref"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, lmbda)
        h_lambda_prima = 1.0 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        Zetas = self.Zetas(lmbda)
        Zetaw = self.Zetaw(lmbda)

        return (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (Zetas + 1.0) + self.XW * Zetaw)
        )
