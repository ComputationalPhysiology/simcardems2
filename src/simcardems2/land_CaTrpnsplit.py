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


# For CaTrpn split


def _XS(XS_prev, XW, gammasu, ksu, kws, dt):
    dXS_dt_linearized = -gammasu - ksu
    dXS_dt = -XS_prev * gammasu - XS_prev * ksu + XW * kws
    return XS_prev + ufl.conditional(
        ufl.gt(abs(dXS_dt_linearized), 1e-8),
        dXS_dt * (dolfin.exp(dXS_dt_linearized * dt) - 1) / dXS_dt_linearized,
        dXS_dt * dt,
    )


def _XW(XW_prev, XU, gammawu, kws, kuw, kwu, dt):
    dXW_dt_linearized = -gammawu - kws - kwu
    dXW_dt = -XW_prev * gammawu - XW_prev * kws + XU * kuw - XW_prev * kwu
    return XW_prev + ufl.conditional(
        ufl.gt(abs(dXW_dt_linearized), 1e-8),
        dXW_dt * (dolfin.exp(dXW_dt_linearized * dt) - 1) / dXW_dt_linearized,
        dXW_dt * dt,
    )


def _TmB(TmB_prev, CaTrpn, XU, ntm, kb, ku, dt):
    dTmB_dt_linearized = -(CaTrpn ** (ntm / 2)) * ku
    dTmB_dt = -TmB_prev * CaTrpn ** (ntm / 2) * ku + XU * (
        kb * ufl.conditional(ufl.lt(CaTrpn ** (-1 / 2 * ntm), 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )

    return TmB_prev + ufl.conditional(
        ufl.gt(abs(dTmB_dt_linearized), 1e-8),
        dTmB_dt * (dolfin.exp(dTmB_dt_linearized * dt) - 1) / dTmB_dt_linearized,
        dTmB_dt * dt,
    )


def _XU(XW, XS, TmB):
    return -XW - XS + 1 - TmB


def _gammawu(Zetaw, gammaw):
    return gammaw * abs(Zetaw)


def _gammasu(Zetas, gammas):
    return gammas * ufl.conditional(
        ufl.gt(Zetas, 0), Zetas, ufl.conditional(ufl.lt(Zetas, -1), -Zetas - 1, 0)
    )


_parameters = {
    "Beta0": 2.3,
    "Tot_A": 25.0,
    "Tref": 120,
    "kuw": 0.182,
    "kws": 0.012,
    "phi": 2.23,
    "rs": 0.25,
    "rw": 0.5,
    "gammas": 0.0085,  # New parameters CaTrpn split
    "gammaw": 0.615,  # New parameters CaTrpn split
    "Trpn50": 0.35,  # New parameters CaTrpn split
    "ntm": 2.4,  # New parameters CaTrpn split.
    "ku": 0.04,  # New parameters CaTrpn split.
}


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        f0,
        s0,
        n0,
        CaTrpn,  # New variables CaTrpn split (missing)
        mesh,
        TmB=None,  # New variables CaTrpn split
        XS=None,
        XW=None,
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
        self.function_space = dolfin.FunctionSpace(mesh, "DG", 1)

        """ For Catrpn split"""
        self.CaTrpn = CaTrpn

        self._XS = dolfin.Function(self.function_space)
        self.XS_prev = dolfin.Function(self.function_space)
        if XS is not None:
            self.XS_prev.assign(XS)

        self._XW = dolfin.Function(self.function_space)
        self.XW_prev = dolfin.Function(self.function_space)
        if XW is not None:
            self.XW_prev.assign(XW)

        self._XU = dolfin.Function(self.function_space)
        self._gammasu = dolfin.Function(self.function_space)
        self._gammawu = dolfin.Function(self.function_space)

        self._TmB = dolfin.Function(self.function_space)
        self.TmB_prev = dolfin.Function(self.function_space)
        if TmB is not None:
            self.TmB_prev.assign(TmB)
        else:  # Set initial TmB value
            self._TmB.interpolate(dolfin.Constant(1))
            self.TmB_prev.interpolate(dolfin.Constant(1))
        """ """

        if parameters is None:
            parameters = _parameters
        self._parameters = parameters

        self._scheme = scheme

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
        self._t_prev = dolfin.Constant(0.0)

    """ For CaTrpn split"""

    @property
    def ksu(self):
        kws = self._parameters["kws"]
        rw = self._parameters["rw"]
        rs = self._parameters["rs"]
        return (kws * rw) * (-1 + 1 / rs)

    @property
    def kwu(self):
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]
        kws = self._parameters["kws"]
        return kuw * (-1 + 1 / rw) - kws

    @property
    def kb(self):
        Trpn50 = self._parameters["Trpn50"]
        ntm = self._parameters["ntm"]
        ku = self._parameters["ku"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        return (Trpn50**ntm * ku) / (-rw * (1 - rs) + 1 - rs)

    @property
    def XW(self):
        return _XW(
            XW_prev=self.XW_prev,
            XU=self.XU,
            gammawu=self.gammawu,
            kws=self._parameters["kws"],
            kuw=self._parameters["kuw"],
            kwu=self.kwu,
            dt=self.dt,
        )

    @property
    def XS(self):
        return _XS(
            XS_prev=self.XS_prev,
            XW=self.XW_prev,
            gammasu=self.gammasu,
            ksu=self.ksu,
            kws=self._parameters["kws"],
            dt=self.dt,
        )

    @property
    def TmB(self):
        return _TmB(
            TmB_prev=self.TmB_prev,
            CaTrpn=self.CaTrpn,
            XU=self.XU,
            ntm=self._parameters["ntm"],
            kb=self.kb,
            ku=self._parameters["ku"],
            dt=self.dt,
        )

    @property
    def XU(self):
        return _XU(
            XW=self.XW_prev,
            XS=self.XS_prev,
            TmB=self.TmB_prev,
        )

    @property
    def gammawu(self):
        return _gammawu(
            Zetaw=self.Zetaw_prev,
            gammaw=self._parameters["gammaw"],
        )

    @property
    def gammasu(self):
        return _gammasu(
            Zetas=self.Zetas_prev,
            gammas=self._parameters["gammas"],
        )

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

    def update_TmB(self):
        logger.debug("update TmB")
        self._projector(
            self._TmB,
            _TmB(
                TmB_prev=self.TmB_prev,
                CaTrpn=self.CaTrpn,
                XU=self.XU,
                ntm=self._parameters["ntm"],
                kb=self.kb,
                ku=self._parameters["ku"],
                dt=self.dt,
            ),
        )

    def update_XS(self):
        logger.debug("update XS")
        self._projector(
            self._XS,
            _XS(
                XS_prev=self.XS_prev,
                XW=self.XW_prev,
                gammasu=self.gammasu,
                ksu=self.ksu,
                kws=self._parameters["kws"],
                dt=self.dt,
            ),
        )

    def update_XW(self):
        logger.debug("update XW")
        self._projector(
            self._XW,
            _XW(
                XW_prev=self.XW_prev,
                XU=self.XU,
                gammawu=self.gammawu,
                kws=self._parameters["kws"],
                kuw=self._parameters["kuw"],
                kwu=self.kwu,
                dt=self.dt,
            ),
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
        return float(self.t - self._t_prev)

    def update_current(self, lmbda):
        self.update_Zetas(lmbda=lmbda)
        self.update_Zetaw(lmbda=lmbda)
        self.update_XS()
        self.update_XW()
        self.update_TmB()

    def update_prev(self):
        logger.debug("update previous")
        self.Zetas_prev.vector()[:] = self._Zetas.vector()
        self.Zetaw_prev.vector()[:] = self._Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda.vector()
        self.XS_prev.vector()[:] = self._XS.vector()
        self.XW_prev.vector()[:] = self._XW.vector()
        self.TmB_prev.vector()[:] = self._TmB.vector()

        self._projector.project(self.Ta_current, self.Ta(self.lmbda))
        self._t_prev.assign(self.t)

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
