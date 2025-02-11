import utils


config = {
    "sim": {
        "N": 2,
        "dt": 0.05,
        "sim_dur": 100.0,
        "mech_mesh": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz",
        "markerfile": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz_surface_ffun",
        "modelfile": "../odefiles/ToRORd_dynCl_endo_caisplit.ode",
        "outdir": "100ms_N1_cai_split_runcheck",
        "split_scheme": "cai",
    },
    "output": {
        "all_ep": ["cai", "v"],
        "all_mech": ["Ta", "lambda"],
        "point_ep": [
            {"name": "cai", "x": 0.0, "y": 0, "z": 0},
            {"name": "v", "x": 0, "y": 0, "z": 0},
        ],
        "point_mech": [
            {"name": "Ta", "x": 0, "y": 0, "z": 0},
            {"name": "Zetas", "x": 0, "y": 0, "z": 0},
            {"name": "XS", "x": 0, "y": 0, "z": 0},
            {"name": "TmB", "x": 0, "y": 0, "z": 0},
            {"name": "CaTrpn", "x": 0, "y": 0, "z": 0},
            {"name": "lambda", "x": 0, "y": 0, "z": 0},
        ],
    },
}

utils.run_3D(config=config)
