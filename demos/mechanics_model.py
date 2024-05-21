import math
import numpy

parameter = {
    "BSLmax": 0,
    "BSRmax": 1,
    "Beta0": 2,
    "Beta1": 3,
    "F": 4,
    "KmBSL": 5,
    "KmBSR": 6,
    "L": 7,
    "R": 8,
    "T": 9,
    "Tot_A": 10,
    "Tref": 11,
    "Trpn50": 12,
    "calib": 13,
    "cat50_ref": 14,
    "cmdnmax": 15,
    "csqnmax": 16,
    "dLambda": 17,
    "emcoupling": 18,
    "etal": 19,
    "etas": 20,
    "gammas": 21,
    "gammaw": 22,
    "isacs": 23,
    "kmcmdn": 24,
    "kmcsqn": 25,
    "kmtrpn": 26,
    "ktrpn": 27,
    "ku": 28,
    "kuw": 29,
    "kws": 30,
    "lmbda": 31,
    "mode": 32,
    "ntm": 33,
    "ntrpn": 34,
    "p_a": 35,
    "p_b": 36,
    "p_k": 37,
    "phi": 38,
    "rad": 39,
    "rs": 40,
    "rw": 41,
    "scale_HF_cat50_ref": 42,
    "trpnmax": 43,
}


def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    return parameter[name]


state = {"Zetaw": 0, "Zetas": 1}


def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    return state[name]


monitor = {
    "Ageo": 0,
    "vcell": 1,
    "Aw": 2,
    "cs": 3,
    "cw": 4,
    "lambda_min12": 5,
    "Acap": 6,
    "vjsr": 7,
    "vmyo": 8,
    "vnsr": 9,
    "vss": 10,
    "As": 11,
    "dZetaw_dt": 12,
    "lambda_min087": 13,
    "dZetas_dt": 14,
    "h_lambda_prima": 15,
    "h_lambda": 16,
    "Ta": 17,
}


def monitor_index(name: str) -> int:
    """Return the index of the monitor with the given name

    Arguments
    ---------
    name : str
        The name of the monitor

    Returns
    -------
    int
        The index of the monitor

    Raises
    ------
    KeyError
        If the name is not a valid monitor
    """

    return monitor[name]


missing = {"XS": 0, "XW": 1}


def missing_index(name: str) -> int:
    """Return the index of the missing with the given name

    Arguments
    ---------
    name : str
        The name of the missing

    Returns
    -------
    int
        The index of the missing

    Raises
    ------
    KeyError
        If the name is not a valid missing
    """

    return missing[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # BSLmax=1.124, BSRmax=0.047, Beta0=2.3, Beta1=-2.4, F=96485.0
    # KmBSL=0.0087, KmBSR=0.00087, L=0.01, R=8314.0, T=310.0
    # Tot_A=25, Tref=120, Trpn50=0.35, calib=1, cat50_ref=0.805
    # cmdnmax=0.05, csqnmax=10.0, dLambda=0, emcoupling=1, etal=200
    # etas=20, gammas=0.0085, gammaw=0.615, isacs=0, kmcmdn=0.00238
    # kmcsqn=0.8, kmtrpn=0.0005, ktrpn=0.1, ku=0.04, kuw=0.182
    # kws=0.012, lmbda=1, mode=1, ntm=2.4, ntrpn=2, p_a=2.1
    # p_b=9.1, p_k=7, phi=2.23, rad=0.0011, rs=0.25, rw=0.5
    # scale_HF_cat50_ref=1.0, trpnmax=0.07

    parameters = numpy.array(
        [
            1.124,
            0.047,
            2.3,
            -2.4,
            96485.0,
            0.0087,
            0.00087,
            0.01,
            8314.0,
            310.0,
            25,
            120,
            0.35,
            1,
            0.805,
            0.05,
            10.0,
            0,
            1,
            200,
            20,
            0.0085,
            0.615,
            0,
            0.00238,
            0.8,
            0.0005,
            0.1,
            0.04,
            0.182,
            0.012,
            1,
            1,
            2.4,
            2,
            2.1,
            9.1,
            7,
            2.23,
            0.0011,
            0.25,
            0.5,
            1.0,
            0.07,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # Zetaw=0, Zetas=0

    states = numpy.array([0, 0], dtype=numpy.float64)

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters, missing_variables):

    # Assign states
    Zetaw = states[0]
    Zetas = states[1]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    F = parameters[4]
    KmBSL = parameters[5]
    KmBSR = parameters[6]
    L = parameters[7]
    R = parameters[8]
    T = parameters[9]
    Tot_A = parameters[10]
    Tref = parameters[11]
    Trpn50 = parameters[12]
    calib = parameters[13]
    cat50_ref = parameters[14]
    cmdnmax = parameters[15]
    csqnmax = parameters[16]
    dLambda = parameters[17]
    emcoupling = parameters[18]
    etal = parameters[19]
    etas = parameters[20]
    gammas = parameters[21]
    gammaw = parameters[22]
    isacs = parameters[23]
    kmcmdn = parameters[24]
    kmcsqn = parameters[25]
    kmtrpn = parameters[26]
    ktrpn = parameters[27]
    ku = parameters[28]
    kuw = parameters[29]
    kws = parameters[30]
    lmbda = parameters[31]
    mode = parameters[32]
    ntm = parameters[33]
    ntrpn = parameters[34]
    p_a = parameters[35]
    p_b = parameters[36]
    p_k = parameters[37]
    phi = parameters[38]
    rad = parameters[39]
    rs = parameters[40]
    rw = parameters[41]
    scale_HF_cat50_ref = parameters[42]
    trpnmax = parameters[43]

    # Assign missing variables
    XS = missing_variables[0]
    XW = missing_variables[1]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)
    vcell = L * (rad * ((3.14 * 1000) * rad))
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    Acap = 2 * Ageo
    vjsr = 0.0048 * vcell
    vmyo = 0.68 * vcell
    vnsr = 0.0552 * vcell
    vss = 0.02 * vcell
    As = Aw
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[0] = dZetaw_dt
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    values[1] = dZetas_dt
    h_lambda_prima = Beta0 * (lambda_min087 + lambda_min12 - 1.87) + 1
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)

    return values


def monitor_values(t, states, parameters, missing_variables):

    # Assign states
    Zetaw = states[0]
    Zetas = states[1]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    F = parameters[4]
    KmBSL = parameters[5]
    KmBSR = parameters[6]
    L = parameters[7]
    R = parameters[8]
    T = parameters[9]
    Tot_A = parameters[10]
    Tref = parameters[11]
    Trpn50 = parameters[12]
    calib = parameters[13]
    cat50_ref = parameters[14]
    cmdnmax = parameters[15]
    csqnmax = parameters[16]
    dLambda = parameters[17]
    emcoupling = parameters[18]
    etal = parameters[19]
    etas = parameters[20]
    gammas = parameters[21]
    gammaw = parameters[22]
    isacs = parameters[23]
    kmcmdn = parameters[24]
    kmcsqn = parameters[25]
    kmtrpn = parameters[26]
    ktrpn = parameters[27]
    ku = parameters[28]
    kuw = parameters[29]
    kws = parameters[30]
    lmbda = parameters[31]
    mode = parameters[32]
    ntm = parameters[33]
    ntrpn = parameters[34]
    p_a = parameters[35]
    p_b = parameters[36]
    p_k = parameters[37]
    phi = parameters[38]
    rad = parameters[39]
    rs = parameters[40]
    rw = parameters[41]
    scale_HF_cat50_ref = parameters[42]
    trpnmax = parameters[43]

    # Assign missing variables
    XS = missing_variables[0]
    XW = missing_variables[1]

    # Assign expressions
    shape = 18 if len(states.shape) == 1 else (18, states.shape[1])
    values = numpy.zeros(shape)
    Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)
    values[0] = Ageo
    vcell = L * (rad * ((3.14 * 1000) * rad))
    values[1] = vcell
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    values[2] = Aw
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    values[3] = cs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    values[4] = cw
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    values[5] = lambda_min12
    Acap = 2 * Ageo
    values[6] = Acap
    vjsr = 0.0048 * vcell
    values[7] = vjsr
    vmyo = 0.68 * vcell
    values[8] = vmyo
    vnsr = 0.0552 * vcell
    values[9] = vnsr
    vss = 0.02 * vcell
    values[10] = vss
    As = Aw
    values[11] = As
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[12] = dZetaw_dt
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    values[13] = lambda_min087
    dZetas_dt = As * dLambda - Zetas * cs
    values[14] = dZetas_dt
    h_lambda_prima = Beta0 * (lambda_min087 + lambda_min12 - 1.87) + 1
    values[15] = h_lambda_prima
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    values[16] = h_lambda
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    values[17] = Ta

    return values


def missing_values(t, states, parameters, missing_variables):

    # Assign states
    Zetaw = states[0]
    Zetas = states[1]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    F = parameters[4]
    KmBSL = parameters[5]
    KmBSR = parameters[6]
    L = parameters[7]
    R = parameters[8]
    T = parameters[9]
    Tot_A = parameters[10]
    Tref = parameters[11]
    Trpn50 = parameters[12]
    calib = parameters[13]
    cat50_ref = parameters[14]
    cmdnmax = parameters[15]
    csqnmax = parameters[16]
    dLambda = parameters[17]
    emcoupling = parameters[18]
    etal = parameters[19]
    etas = parameters[20]
    gammas = parameters[21]
    gammaw = parameters[22]
    isacs = parameters[23]
    kmcmdn = parameters[24]
    kmcsqn = parameters[25]
    kmtrpn = parameters[26]
    ktrpn = parameters[27]
    ku = parameters[28]
    kuw = parameters[29]
    kws = parameters[30]
    lmbda = parameters[31]
    mode = parameters[32]
    ntm = parameters[33]
    ntrpn = parameters[34]
    p_a = parameters[35]
    p_b = parameters[36]
    p_k = parameters[37]
    phi = parameters[38]
    rad = parameters[39]
    rs = parameters[40]
    rw = parameters[41]
    scale_HF_cat50_ref = parameters[42]
    trpnmax = parameters[43]

    # Assign missing variables
    XS = missing_variables[0]
    XW = missing_variables[1]

    # Assign expressions
    shape = 2 if len(states.shape) == 1 else (2, states.shape[1])
    values = numpy.zeros(shape)
    values[0] = Zetas
    values[1] = Zetaw
    Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)

    return values


def forward_generalized_rush_larsen(states, t, dt, parameters, missing_variables):

    # Assign states
    Zetaw = states[0]
    Zetas = states[1]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    F = parameters[4]
    KmBSL = parameters[5]
    KmBSR = parameters[6]
    L = parameters[7]
    R = parameters[8]
    T = parameters[9]
    Tot_A = parameters[10]
    Tref = parameters[11]
    Trpn50 = parameters[12]
    calib = parameters[13]
    cat50_ref = parameters[14]
    cmdnmax = parameters[15]
    csqnmax = parameters[16]
    dLambda = parameters[17]
    emcoupling = parameters[18]
    etal = parameters[19]
    etas = parameters[20]
    gammas = parameters[21]
    gammaw = parameters[22]
    isacs = parameters[23]
    kmcmdn = parameters[24]
    kmcsqn = parameters[25]
    kmtrpn = parameters[26]
    ktrpn = parameters[27]
    ku = parameters[28]
    kuw = parameters[29]
    kws = parameters[30]
    lmbda = parameters[31]
    mode = parameters[32]
    ntm = parameters[33]
    ntrpn = parameters[34]
    p_a = parameters[35]
    p_b = parameters[36]
    p_k = parameters[37]
    phi = parameters[38]
    rad = parameters[39]
    rs = parameters[40]
    rw = parameters[41]
    scale_HF_cat50_ref = parameters[42]
    trpnmax = parameters[43]

    # Assign missing variables
    XS = missing_variables[0]
    XW = missing_variables[1]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)
    vcell = L * (rad * ((3.14 * 1000) * rad))
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    Acap = 2 * Ageo
    vjsr = 0.0048 * vcell
    vmyo = 0.68 * vcell
    vnsr = 0.0552 * vcell
    vss = 0.02 * vcell
    As = Aw
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    dZetaw_dt_linearized = -cw
    values[0] = Zetaw + numpy.where(
        (numpy.abs(dZetaw_dt_linearized) > 1e-08),
        dZetaw_dt * (numpy.exp(dZetaw_dt_linearized * dt) - 1) / dZetaw_dt_linearized,
        dZetaw_dt * dt,
    )
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    dZetas_dt_linearized = -cs
    values[1] = Zetas + numpy.where(
        (numpy.abs(dZetas_dt_linearized) > 1e-08),
        dZetas_dt * (numpy.exp(dZetas_dt_linearized * dt) - 1) / dZetas_dt_linearized,
        dZetas_dt * dt,
    )
    h_lambda_prima = Beta0 * (lambda_min087 + lambda_min12 - 1.87) + 1
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)

    return values
