import pulse
import dolfin
import ufl_legacy as ufl
import logging
import numpy as np
from enum import Enum

from simcardems2 import utils

logger = logging.getLogger(__name__)


class Scheme(str, Enum):
    fd = "fd"
    bd = "bd"
    analytic = "analytic" # Currently the only scheme. Add fd and bd?


def _Zeta(Zeta, A, c, dLambda, dt, scheme: Scheme): # Note: dLambda = lambda - prev_lambda, so different from in .ode!
    if scheme == Scheme.analytic:
        if dt == 0:
            print("NB!: dt equal to zero in forward step! This is expected for initialisation step")
            print("dLambda:", dLambda)
            return Zeta
        else:            
            return (np.where(
                (np.abs(-c) > 1e-08),
                Zeta * np.exp(-c * dt) + (A * dLambda / (c * dt)) * (1.0 - np.exp(-c * dt)), 
                Zeta + A*dLambda - Zeta*c*dt
                )
            )

""" New variables CaTrpn split, explicit updates"""
def _XS(XS, XW, gammasu, ksu, kws, dt, scheme: Scheme):
    if scheme == Scheme.analytic:  
        return (XS + np.where((np.abs(-gammasu - ksu) > 1e-08),
                (-XS * gammasu - XS * ksu + XW * kws) * (np.exp((-gammasu - ksu) * dt) - 1) / (-gammasu - ksu),
                (-XS * gammasu - XS * ksu + XW * kws) * dt)
                )

    else:
        print("_XS not updating. Change or implement new scheme")

def _XW(XS, XW, XU, gammawu, kws, kuw, kwu, dt, scheme: Scheme):
    if scheme == Scheme.analytic:  
        return (XW + np.where(
                (np.abs(-gammawu - kws - kwu) > 1e-08),
                (-XW * gammawu - XW * kws + XU * kuw - XW * kwu) * (np.exp((-gammawu - kws - kwu) * dt) - 1) / (-gammawu - kws - kwu),
                (-XW * gammawu - XW * kws + XU * kuw - XW * kwu) * dt
                )
            )    
    else:
        print("_XW not updating. Change or implement new scheme")
               
def _TmB(TmB, CaTrpn, XU, ntm, kb, ku, dt, scheme: Scheme):
    if scheme == Scheme.analytic:  
        return (TmB + np.where(
                (np.abs(-(CaTrpn ** (ntm / 2)) * ku) > 1e-08),
                (-TmB * CaTrpn ** (ntm / 2) * ku + XU * (
                    kb
                    * np.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
                )) * (np.exp((-(CaTrpn ** (ntm / 2)) * ku) * dt) - 1) / (-(CaTrpn ** (ntm / 2)) * ku),
                (-TmB * CaTrpn ** (ntm / 2) * ku + XU * (
                    kb
                    * np.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
                )) * dt
            )
        )
    else:
        print("_TmB not updating. Change or implement new scheme")

# Monitors    
def _XU(XW, XS, TmB):
    return -XW -XS + 1 - TmB

def _gammasu(Zetas, gammas):
    return gammas * np.where(
        (Zetas > 0), 
        Zetas, 
        np.where((Zetas < -1), 
        -Zetas - 1, 0)
        )  

def _gammawu(Zetaw, gammaw):
    return gammaw * np.abs(Zetaw)
    
""" """


_parameters = {
    "Beta0": 2.3,
    "Tot_A": 25.0,
    "Tref": 120,
    "kuw": 0.182, # 0.026 in land 2017 ?
    "kws": 0.012, # 0.004 in land 2017?
    "phi": 2.23,
    "rs": 0.25,
    "rw": 0.5,
    "gammas": 0.0085,  # New parameters CaTrpn split
    "gammaw": 0.615,  # New parameters CaTrpn split
    "Trpn50": 0.35,   # New parameters CaTrpn split
    "ntm": 2.4, # New parameters CaTrpn split. # TODO: check. ntm = 2.2 or 5 in Land et al. 2017 vs 2.4 in .ode files
    "ku": 0.04, # New parameters CaTrpn split. # TODO: check. ku = 1 in Land et al. 2017
}


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        f0,
        s0,
        n0,
        CaTrpn,  # New variables CaTrpn split (missing)
        mesh,
        TmB=None, # New variables CaTrpn split 
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
        self.function_space = dolfin.FunctionSpace(mesh, "CG", 1)

        """ New variables CaTrpn split """
        self.CaTrpn = CaTrpn
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
            
        """ New variables CaTrpn split """
        
        self._XU = dolfin.Function(self.function_space)
        self._gammasu = dolfin.Function(self.function_space)
        self._gammawu = dolfin.Function(self.function_space)
        
        
        self._XS = dolfin.Function(self.function_space)
        self.XS_prev = dolfin.Function(self.function_space)
        if XS is not None:
            self.XS_prev.assign(XS)

        self._XW = dolfin.Function(self.function_space)
        self.XW_prev = dolfin.Function(self.function_space)
        if XW is not None:
            self.XW_prev.assign(XW)
            
        self._TmB = dolfin.Function(self.function_space)
        self.TmB_prev = dolfin.Function(self.function_space)
        if TmB is not None:
            self.TmB_prev.assign(TmB)
        else: # Set initial TmB value
            self._TmB.interpolate(dolfin.Constant(1))
            self.TmB_prev.interpolate(dolfin.Constant(1))
        """ """
            

        self.Ta_current = dolfin.Function(self.function_space, name="Ta")
        self._projector = utils.Projector(self.function_space)
        self._dLambda_tol = dLambda_tol
        self._t_prev = 0.0

    """ New variables CaTrpn split """
    @property
    def ksu(self):
        kws = self._parameters["kws"]
        rw = self._parameters["rw"]
        rs = self._parameters["rs"]
        return ((kws * rw) * (-1 + 1 / rs))
    
    @property
    def kwu(self):
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]
        kws = self._parameters["kws"]
        return (kuw * (-1 + 1 / rw) - kws)
    
    @property
    def kb(self):
        Trpn50 = self._parameters["Trpn50"]
        ntm = self._parameters["ntm"]
        ku = self._parameters["ku"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        return ((Trpn50**ntm * ku) / (-rw * (1 - rs) + 1 - rs))
    """ """


    @property
    def dLambda(self):
        logger.debug("Evaluate dLambda")
        self._dLambda.vector()[:] = self.lmbda.vector() - self.lmbda_prev.vector()
        self._dLambda.vector()[
            np.where(np.abs(self._dLambda.vector().get_local()) < self._dLambda_tol)[0]
        ] = 0.0
        return self._dLambda

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

    def update_Zetas(self):
        logger.debug("update Zetas")
        self._Zetas.vector()[:] = _Zeta(
            #self.Zetas_prev.vector(),
            self.Zetas_prev.vector().get_local(),
            self.As,
            self.cs,
            #self.dLambda.vector(),
            self.dLambda.vector().get_local(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetas(self):
        return self._Zetas

    def update_Zetaw(self):
        logger.debug("update Zetaw")
        self._Zetaw.vector()[:] = _Zeta(
            #self.Zetaw_prev.vector(),
            self.Zetaw_prev.vector().get_local(),
            self.Aw,
            self.cw,
            #self.dLambda.vector(),
            self.dLambda.vector().get_local(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetaw(self):
        return self._Zetaw


    """ New variables CaTrpn split"""
    @property
    def XS(self):
        return self._XS
    
    @property
    def XW(self):
        return self._XW
    
    @property
    def TmB(self):
        return self._TmB
    
    @property
    def XU(self):
        return self._XU
    
    @property
    def gammawu(self):
        return self._gammawu
    
    @property
    def gammasu(self):
        return self._gammasu
    
    
    def update_TmB(self):
        logger.debug("update TmB")
        self._TmB.vector()[:] = _TmB(
            self.TmB_prev.vector().get_local(),
            self.CaTrpn.vector().get_local(), #missing variable
            self.XU.vector().get_local(),
            self._parameters["ntm"],
            self.kb,
            self._parameters["ku"],
            self.dt,
            self._scheme,
            )
    
    def update_XS(self):
        logger.debug("update XS") 
        self._XS.vector()[:] = _XS(
            self.XS_prev.vector().get_local(),
            self.XW_prev.vector().get_local(),
            self.gammasu.vector().get_local(),
            self.ksu,
            self._parameters["kws"],
            self.dt,
            self._scheme,
            )
                
    def update_XW(self):
        logger.debug("update XW")
        self._XW.vector()[:] = _XW(
            self.XS_prev.vector().get_local(),
            self.XW_prev.vector().get_local(),
            self.XU.vector().get_local(),
            self.gammawu.vector().get_local(),
            self._parameters["kws"],
            self._parameters["kuw"],
            self.kwu,
            self.dt,
            self._scheme,
            )
        
    # Calculate monitors
    def calculate_XU(self):
        logger.debug("update XU")
        self._XU.vector()[:] = _XU(
            self.XW_prev.vector().get_local(),
            self.XS_prev.vector().get_local(),
            self.TmB_prev.vector().get_local(),
            )
        
    def calculate_gammasu(self):
        logger.debug("update gammasu")
        self._gammasu.vector()[:] = _gammasu(
            self.Zetas_prev.vector().get_local(),
            self._parameters["gammas"],
            )
        
    def calculate_gammawu(self):
        logger.debug("update gammawu")
        self._gammawu.vector()[:] = _gammawu(
            self.Zetaw_prev.vector().get_local(),
            self._parameters["gammaw"],
            )
        
    """ """


    @property
    def dt(self) -> float:
        return self.t - self._t_prev


    def update_prev(self):
        logger.debug("update previous")
        self.Zetas_prev.vector()[:] = self.Zetas.vector()
        self.Zetaw_prev.vector()[:] = self.Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda.vector()
        
        """ New variables CaTrpn split"""
        self.XS_prev.vector()[:] = self.XS.vector()
        self.XW_prev.vector()[:] = self.XW.vector()
        self.TmB_prev.vector()[:] = self.TmB.vector()
        """ """
        
        self._projector.project(self.Ta_current, self.Ta)
        self._t_prev = self.t

    @property
    def Ta(self):
        logger.debug("Evaluate Ta")
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = 1.0  # self._parameters["scale_popu_Tref"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(self.lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, self.lmbda)
        h_lambda_prima = 1.0 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        return (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (self.Zetas + 1.0) + self.XW * self.Zetaw)
        )
        #return ( # To be consistent with 0D
        #    h_lambda
        #    * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
        #    * (self.XS_prev * (self.Zetas_prev + 1.0) + self.XW_prev * self.Zetaw_prev)
        #)

    def Wactive(self, F, **kwargs): 
        # Overrides Wactive of pulse.ActiveModel
        # and is used in HolzapfelOgden to calculate strain_energy
        """Active stress energy"""
        logger.debug("Compute active stress energy")
        C = F.T * F
        C = F.T * F
        f = F * self.f0
        self._projector.project(self.lmbda, dolfin.sqrt(f**2))
        self.update_Zetas()
        self.update_Zetaw()
        """ New variables to update in CaTrpn split"""
        self.calculate_XU()
        self.calculate_gammasu()
        self.calculate_gammawu()
        
        self.update_XS()
        self.update_XW()
        self.update_TmB()        
        """ """
        return pulse.material.active_model.Wactive_transversally(
            Ta=self.Ta,
            C=C,
            f0=self.f0,
            eta=self.eta,
        )
