def validate_input_types(config):
    """Check that parameters provided in the config file are of correct type"""
    # TODO: check for missing parameters and provide defaults

    number_par = [
        ("sim", "sim_dur"),
        ("sim", "dt"),
        ("sim", "N"),
        ("ep", "sigma_il"),
        ("ep", "sigma_it"),
        ("ep", "sigma_el"),
        ("ep", "sigma_et"),
        ("mech", "a"),
        ("mech", "a_f"),
        ("mech", "b"),
        ("mech", "b_f"),
        ("mech", "a_s"),
        ("mech", "b_s"),
        ("mech", "a_fs"),
        ("mech", "b_fs"),
        ("stim", "start"),
        ("stim", "amplitude"),
        ("stim", "duration"),
        ("stim", "xmin"),
        ("stim", "xmax"),
        ("stim", "ymin"),
        ("stim", "ymax"),
        ("stim", "zmin"),
        ("stim", "zmax"),
    ]
    int_par = [
        ("sim", "N"),
        ("write_all_ep", "numbers"),
        ("write_all_mech", "numbers"),
        ("bcs", "numbers"),
    ]
    str_par = [
        ("sim", "mech_mesh"),
        ("sim", "outdir"),
        ("sim", "modelfile"),
        ("bcs", "markerfile"),
    ]

    for section, param in number_par:
        assert isinstance(config[section][param], (int, float)), (
            f"Parameter '{section}.{param} = {config[section][param]}' "
            f"is a {type(config[section][param]).__name__}. "
            "Provide an integer or a float"
        )

    for section, param in int_par:
        assert isinstance(config[section][param], (int)), (
            f"Parameter '{section}.{param} = {config[section][param]}' "
            f"is a {type(config[section][param]).__name__}. "
            "Provide an integer"
        )

    for section, param in str_par:
        assert isinstance(config[section][param], (str)), (
            f"Parameter '{section}.{param} = {config[section][param]}' "
            f"is a {type(config[section][param]).__name__}. "
            "Provide a string"
        )

    # Validate parameters for writing output
    write_var = [
        ("write_all_ep", "numbers"),
        ("write_all_mech", "numbers"),
        ("write_point_ep", "numbers"),
        ("write_point_mech", "numbers"),
    ]
    write_var_coords = [
        "write_point_ep",
        "write_point_mech",
    ]

    bcs_types = ["Dirichlet", "Neumann"]
    bcs_func_spaces = ["u_x", "u_y", "u_z"]
    bcs_int_pars = ["marker", "param_numbers"]

    for section, param in write_var:
        for i in range(config[section][param]):
            name = config[section][str(i)]["name"]
            assert isinstance(name, str), (
                f"Parameter '{section}.{i}.name = {name}' is a "
                f"{type(name).__name__}. Provide a string"
            )

            if section in write_var_coords:
                for coord in ["x", "y", "z"]:
                    coord_value = config[section][str(i)][coord]
                    assert isinstance(coord_value, (int, float)), (
                        f"Parameter '{section}.{i}.{coord} = {coord_value}' is a "
                        f"{type(coord_value).__name__}. Provide an integer or float."
                    )

    for bcs_nr in range(config["bcs"]["numbers"]):
        for bcs_int_par in bcs_int_pars:
            bcs_int_value = config["bcs"][str(bcs_nr)][bcs_int_par]
            assert isinstance(bcs_int_value, int), (
                f"Parameter 'bcs.{bcs_nr}.{bcs_int_par} = {bcs_int_value}' is a "
                f"{type(bcs_int_value).__name__}. Provide an integer."
            )

        bcs_type = config["bcs"][str(bcs_nr)]["type"]
        assert (
            bcs_type in bcs_types
        ), f"Parameter 'bcs.{bcs_nr}.type = {bcs_type}' must be one of {bcs_types}."

        if bcs_type == "Dirichlet":
            bcs_V = config["bcs"][str(bcs_nr)]["V"]
            assert (
                bcs_V in bcs_func_spaces
            ), f"Parameter 'bcs.{bcs_nr}.V = {bcs_V}' must be one of {bcs_func_spaces}."

        for param_nr in range(config["bcs"][str(bcs_nr)]["param_numbers"]):
            param_name = config["bcs"][str(bcs_nr)]["param"][str(param_nr)]["name"]
            assert isinstance(param_name, str), (
                f"Parameter 'bcs.{bcs_nr}.param.{param_nr}.name = {param_name}' is a "
                f"{type(param_name).__name__}. Provide a string."
            )
            # When bcs expression is given instead of constant, check the "degree" type
            bcs_int_value = config["bcs"][str(bcs_nr)]["degree"]
            assert isinstance(bcs_int_value, int), (
                f"Parameter 'bcs.{bcs_nr}.degree = {bcs_int_value}' is a "
                f"{type(bcs_int_value).__name__}. Provide an integer."
            )

            param_value = config["bcs"][str(bcs_nr)]["param"][str(param_nr)]["value"]
            assert isinstance(param_value, (int, float)), (
                f"Parameter 'bcs.{bcs_nr}.param.{param_nr}.value = {param_value}' is a "
                f"{type(param_value).__name__}. Provide an integer or float."
            )

    print("Input types validated")
