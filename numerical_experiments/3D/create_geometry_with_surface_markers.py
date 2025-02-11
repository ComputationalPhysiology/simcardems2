#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:47:22 2024

@author: lenamyklebust
"""

import argparse
import numpy as np
import dolfin
from pathlib import Path
from typing import Dict
from typing import Optional


def setup_geometry(dx: float, Lx: float, Ly: float, Lz: float) -> dolfin.Mesh:
    mesh = dolfin.BoxMesh(
        dolfin.MPI.comm_world,
        dolfin.Point(0.0, 0.0, 0.0),
        dolfin.Point(Lx, Ly, Lz),
        int(np.rint((Lx / dx))),
        int(np.rint((Ly / dx))),
        int(np.rint((Lz / dx))),
    )
    return mesh


def default_markers() -> Dict[str, int]:
    return {
        "X0": 1,
        "X1": 2,
        "Y0": 3,
        "Y1": 4,
        "Z0": 5,
        "Z1": 6,
    }


def assign_surface_markers(
    mesh: dolfin.Mesh,
    lx: float,
    ly: float,
    lz: float,
    markers: Optional[Dict[str, int]] = None,
) -> dolfin.MeshFunction:
    if markers is None:
        markers = default_markers()

    # Define domains for each surface
    x0 = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    x1 = dolfin.CompiledSubDomain("near(x[0], lx) && on_boundary", lx=lx)
    y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    y1 = dolfin.CompiledSubDomain("near(x[1], ly) && on_boundary", ly=ly)
    z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")
    z1 = dolfin.CompiledSubDomain("near(x[2], lz) && on_boundary", lz=lz)

    ffun = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ffun.set_all(0)

    x0.mark(ffun, markers["X0"])
    x1.mark(ffun, markers["X1"])

    y0.mark(ffun, markers["Y0"])
    y1.mark(ffun, markers["Y1"])

    z0.mark(ffun, markers["Z0"])
    z1.mark(ffun, markers["Z1"])
    return ffun, markers


def main(outdir: Path, dx: float = 0.5, Lx: float = 0.5, Ly: float = 1.0, Lz: float = 2.0):
    print(
        f"Creating geometry, mesh and surface markers "
        f"with dx={dx}, Lx={Lx}, Ly={Ly}, Lz={Lz} in {outdir}"
    )
    outdir.mkdir(exist_ok=True, parents=True)
    mesh = setup_geometry(dx=dx, Lx=Lx, Ly=Ly, Lz=Lz)
    mesh_file = outdir / f"mesh_mech_{dx}dx_{Lx}Lx_{Ly}Ly_{Lz}Lz.xdmf"
    mesh_file.unlink(missing_ok=True)
    mesh_file.with_suffix(".h5").unlink(missing_ok=True)

    surface_ffun, surface_markers = assign_surface_markers(mesh, lx=Lx, ly=Ly, lz=Lz)
    surface_func_file = outdir / f"mesh_mech_{dx}dx_{Lx}Lx_{Ly}Ly_{Lz}Lz_surface_ffun.xdmf"
    surface_func_file.unlink(missing_ok=True)
    surface_func_file.with_suffix(".h5").unlink(missing_ok=True)

    with dolfin.XDMFFile(mesh_file.as_posix()) as file:
        file.write(mesh)
    with dolfin.XDMFFile(surface_func_file.as_posix()) as xdmf:
        xdmf.write(surface_ffun)
    print("Saved mesh and surface markers to", mesh_file, surface_func_file)
    print(f"Surface markers: {surface_markers}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=Path, default=Path("meshes"))
    parser.add_argument("--dx", type=float, default=0.5)
    parser.add_argument("--Lx", type=float, default=0.5)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--Lz", type=float, default=2.0)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
