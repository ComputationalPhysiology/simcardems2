from dataclasses import dataclass
import dolfin
import numpy as np
import ufl_legacy as ufl


@dataclass
class MissingValue:
    element: ufl.FiniteElementBase
    interpolation_element: ufl.FiniteElementBase
    mechanics_mesh: dolfin.Mesh
    ep_mesh: dolfin.Mesh
    num_values: int

    def __post_init__(self):
        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, self.element)
        self.V_mechanics = dolfin.FunctionSpace(self.mechanics_mesh, self.element)
        self.V_ep_int = dolfin.FunctionSpace(self.ep_mesh, self.interpolation_element)
        self.V_mechanics_int = dolfin.FunctionSpace(self.mechanics_mesh, self.interpolation_element)

        self.u_ep = [dolfin.Function(self.V_ep) for _ in range(self.num_values)]
        self.u_mechanics = [dolfin.Function(self.V_mechanics) for _ in range(self.num_values)]

        self.u_ep_int = [dolfin.Function(self.V_ep_int) for _ in range(self.num_values)]
        self.u_mechanics_int = [
            dolfin.Function(self.V_mechanics_int) for _ in range(self.num_values)
        ]

        self.values_ep = np.zeros((self.num_values, self.u_ep[0].vector().local_size()))
        self.values_mechanics = np.zeros(
            (self.num_values, self.u_mechanics[0].vector().local_size())
        )

    def ep_values_to_function(self) -> None:
        for i in range(self.num_values):
            self.u_ep[i].vector().set_local(self.values_ep[i])

    def ep_function_to_values(self) -> None:
        for i in range(self.num_values):
            self.values_ep[i, :] = self.u_ep[i].vector().get_local()

    def mechanics_values_to_function(self) -> None:
        for i in range(self.num_values):
            self.u_mechanics[i].vector().set_local(self.values_mechanics[i])

    def mechanics_function_to_values(self) -> None:
        for i in range(self.num_values):
            self.values_mechanics[i, :] = self.u_mechanics[i].vector().get_local()

    def interpolate_ep_to_mechanics(self) -> None:
        for i in range(self.num_values):
            self.u_mechanics[i].interpolate(self.u_ep_int[i])

    def interpolate_mechanics_to_ep(self) -> None:
        for i in range(self.num_values):
            self.u_ep[i].interpolate(self.u_mechanics_int[i])
