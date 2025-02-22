from pathlib import Path
import time
import json

from typing import NamedTuple, Literal, Sequence


import numpy as np
import matplotlib.pyplot as plt
import gotranx


class TrackedValue(NamedTuple):
    ep_value: np.ndarray
    mechanics_value: np.ndarray
    full_value: np.ndarray
    ep_type: Literal["state", "monitor", "parameter", "none"]
    mechanics_type: Literal["state", "monitor", "parameter", "none"]
    full_type: Literal["state", "monitor", "parameter", "none"] = "none"
    full_index: int = -1
    ep_index: int = -1
    mechanics_index: int = -1


def update_tracked_values(
    index: int,
    tracked_values: dict[str, TrackedValue],
    model: Literal["mechanics", "ep", "full"],
    y: np.ndarray,
    monitor: np.ndarray,
    p: np.ndarray,
) -> None:
    for name, v in tracked_values.items():
        if model == "mechanics":
            if v.mechanics_type == "state":
                v.mechanics_value[index] = y[v.mechanics_index]
            elif v.mechanics_type == "monitor":
                v.mechanics_value[index] = monitor[v.mechanics_index]
            elif v.mechanics_type == "parameter":
                v.mechanics_value[index] = p[v.mechanics_index]
        elif model == "ep":
            if v.ep_type == "state":
                v.ep_value[index] = y[v.ep_index]
            elif v.ep_type == "monitor":
                v.ep_value[index] = monitor[v.ep_index]
            elif v.ep_type == "parameter":
                v.ep_value[index] = p[v.ep_index]
        elif model == "full":
            if v.full_type == "state":
                v.full_value[index] = y[v.full_index]
            elif v.full_type == "monitor":
                v.full_value[index] = monitor[v.full_index]
            elif v.full_type == "parameter":
                v.full_value[index] = p[v.full_index]


def setup_track_values(
    mechanics_model, ep_model, full_model, track_names, N
) -> dict[str, TrackedValue]:
    track_values = {}
    for name in track_names:
        if name in mechanics_model["state"]:
            mechanics_index = mechanics_model["state_index"](name)
            mechanics_type = "state"
        elif name in mechanics_model["monitor"]:
            mechanics_index = mechanics_model["monitor_index"](name)
            mechanics_type = "monitor"
        elif name in mechanics_model["parameter"]:
            mechanics_index = mechanics_model["parameter_index"](name)
            mechanics_type = "parameter"
        else:
            mechanics_index = -1
            mechanics_type = "none"

        if name in ep_model["state"]:
            ep_index = ep_model["state_index"](name)
            ep_type = "state"
        elif name in ep_model["monitor"]:
            ep_index = ep_model["monitor_index"](name)
            ep_type = "monitor"
        elif name in ep_model["parameter"]:
            ep_index = ep_model["parameter_index"](name)
            ep_type = "parameter"
        else:
            ep_index = -1
            ep_type = "none"

        if name in full_model["state"]:
            full_index = full_model["state_index"](name)
            full_type = "state"
        elif name in full_model["monitor"]:
            full_index = full_model["monitor_index"](name)
            full_type = "monitor"
        elif name in full_model["parameter"]:
            full_index = full_model["parameter_index"](name)
            full_type = "parameter"
        else:
            full_index = -1
            full_type = "none"

        track_values[name] = TrackedValue(
            ep_value=np.zeros(N),
            mechanics_value=np.zeros(N),
            full_value=np.zeros(N),
            ep_type=ep_type,
            mechanics_type=mechanics_type,
            full_type=full_type,
            full_index=full_index,
            ep_index=ep_index,
            mechanics_index=mechanics_index,
        )
    return track_values


class MissingValue(NamedTuple):
    name: str  # Name of the missing value
    model: Literal["mechanics", "ep"]  # Which model is missing the value
    index: int  # Index of the missing value in the other model
    value: float  # Default value of the missing value
    type: Literal["state", "monitor"]  # Type of the missing value


class MissingValueIndices(NamedTuple):
    ep: list[MissingValue]
    mechanics: list[MissingValue]

    @property
    def mechanics_values(self):
        return np.array([v.value for v in self.mechanics])

    @property
    def ep_values(self):
        return np.array([v.value for v in self.ep])


def setup_missing_values(mechanics_model, ep_model) -> MissingValueIndices:
    missing_values_mech = []
    for k in mechanics_model.get("missing", {}):
        if k in ep_model["state"]:
            index = ep_model["state_index"](k)
            type = "state"
            value = ep_model["init_state_values"]()[index]
        elif k in ep_model["monitor"]:
            index = ep_model["monitor_index"](k)
            type = "monitor"
            value = 0.0  # FIXME
        else:
            raise ValueError(f"Missing variable {k} not found in state or monitor")

        missing_values_mech.append(
            MissingValue(name=k, model="mechanics", index=index, value=value, type=type)
        )

    missing_values_ep = []
    for k in ep_model.get("missing", {}):
        if k in mechanics_model["state"]:
            index = mechanics_model["state_index"](k)
            type = "state"
            value = mechanics_model["init_state_values"]()[index]
        elif k in mechanics_model["monitor"]:
            index = mechanics_model["monitor_index"](k)
            type = "monitor"
            value = 0.0  # FIXME
        else:
            raise ValueError(f"Missing variable {k} not found in state or monitor")
        missing_values_ep.append(
            MissingValue(name=k, model="ep", index=index, value=value, type=type)
        )

    return MissingValueIndices(ep=missing_values_ep, mechanics=missing_values_mech)


def twitch(t, tstart=0.05, ca_ampl=-0.2):
    tau1 = 0.05 * 1000
    tau2 = 0.110 * 1000

    ca_diast = 0.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1) - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca + 1.0


def plot_twitch(t, filename="twitch.png"):
    fig, ax = plt.subplots()
    ax.plot(t, twitch(t))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Lambda")
    fig.savefig(filename)


def run_0D(
    odefile: Path,
    outdir: Path,
    track_names: Sequence[str],
    Ns: Sequence[int],
    save_traces: bool = False,
    run_full_model: bool = False,
    dt: float = 0.05,
    simdur: float = 10.0,
):
    outdir.mkdir(exist_ok=True)

    # Load the model
    ode = gotranx.load_ode(odefile)

    mechanics_comp = ode.get_component("mechanics")
    mechanics_ode = mechanics_comp.to_ode()

    ep_ode = ode - mechanics_comp
    ep_file = Path(f"{odefile.stem}_ep.py")
    mechanics_file = Path(f"{odefile.stem}_mechanics.py")
    full_file = Path(f"{odefile.stem}.py")

    # Generate model code from .ode file
    rebuild = False
    if not ep_file.is_file() or rebuild:
        # Generate code for full model.
        code = gotranx.cli.gotran2py.get_code(
            ode,
            scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
        )

        # Generate code for the electrophysiology model
        code_ep = gotranx.cli.gotran2py.get_code(
            ep_ode,
            scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
            missing_values=mechanics_ode.missing_variables,
        )

        # Generate code for the mechanics model
        code_mechanics = gotranx.cli.gotran2py.get_code(
            mechanics_ode,
            scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
            missing_values=ep_ode.missing_variables,
        )

        # Create ep, mechanics and full model to files:
        ep_file.write_text(code_ep)
        mechanics_file.write_text(code_mechanics)
        full_file.write_text(code)

    # Import ep, mechanics and full model

    model = __import__(full_file.stem).__dict__
    ep_model = __import__(ep_file.stem).__dict__
    mechanics_model = __import__(mechanics_file.stem).__dict__

    t = np.arange(0, simdur, dt)

    # Forwared generalized rush larsen scheme for the electrophysiology model
    fgr_ep = ep_model["forward_generalized_rush_larsen"]
    # Monitor function for the electrophysiology model
    mon_ep = ep_model["monitor_values"]
    # Missing values function for the electrophysiology model
    mv_ep = ep_model["missing_values"]

    # Forwared generalized rush larsen scheme for the mechanics model
    fgr_mechanics = mechanics_model["forward_generalized_rush_larsen"]
    # Monitor function for the mechanics model
    mon_mechanics = mechanics_model["monitor_values"]
    # Missing values function for the mechanics model
    mv_mechanics = mechanics_model["missing_values"]

    # Setup the track values
    track_values = setup_track_values(
        mechanics_model=mechanics_model,
        ep_model=ep_model,
        full_model=model,
        track_names=track_names,
        N=len(t),
    )
    missing_value_indices = setup_missing_values(mechanics_model=mechanics_model, ep_model=ep_model)

    # Index of the stretch and stretch rate in the mechanics model
    lmbda_index_mechanics = mechanics_model["parameter_index"]("lmbda")
    dLambda_index_mechanics = mechanics_model["parameter_index"]("dLambda")

    # Forwared generalized rush larsen scheme for the full model
    fgr = model["forward_generalized_rush_larsen"]
    # Monitor function for the full model
    mon = model["monitor_values"]
    # Index of the stretch and stretch rate in the full model
    lmbda_index = model["parameter_index"]("lmbda")
    dLambda_index = model["parameter_index"]("dLambda")
    
    
    # lambda for use on EP side
    if "dLambda" in ep_model['parameter']:
        dLambda_index_ep = ep_model["parameter_index"]("dLambda")
    if "lmbda" in ep_model['parameter']:
        lmbda_index_ep = ep_model["parameter_index"]("lmbda")
    
    for N in Ns:
        timing_init = time.perf_counter()
        # Get initial values from the EP model
        y_ep = ep_model["init_state_values"]()
        p_ep = ep_model["init_parameter_values"]()
        ep_missing_values = np.zeros(len(ep_ode.missing_variables))
        if len(ep_missing_values) > 0:
            ep_missing_args = (ep_missing_values,)
        else:
            ep_missing_args = ()

        # Get initial values from the mechanics model
        y_mechanics = mechanics_model["init_state_values"]()
        p_mechanics = mechanics_model["init_parameter_values"]()

        mechanics_missing_values = missing_value_indices.mechanics_values

        # Get the initial values from the full model
        y = model["init_state_values"]()
        p = model["init_parameter_values"]()  # Used in lambda update

        # Get the default values of the missing values
        # A little bit chicken and egg problem here, but in this specific case we know that
        # the mechanics_missing_values is only the calcium concentration, which is a state variable
        # and this doesn't require any additional information to be calculated.
        mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, *ep_missing_args)
        if ep_missing_args:
            ep_missing_values[:] = mv_mechanics(
                0, y_mechanics, p_mechanics, mechanics_missing_values
            )

        # We will store the previous missing values to check for convergence and use for updating
        prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
        prev_mechanics_missing_values[:] = mechanics_missing_values

        inds = []
        count = 1
        prev_lmbda = p[lmbda_index]

        timings_solveloop = []
        timings_ep_steps = []
        timings_mech_steps = []
        for i, ti in enumerate(t):
            timing_loopstart = time.perf_counter()
            # Set initial lambda
            if ti == 0:
                lmbda_ti = twitch(ti)
                p[lmbda_index] = lmbda_ti
                p_mechanics[lmbda_index_mechanics] = lmbda_ti
                dLambda = 0
                p[dLambda_index] = dLambda
                p_mechanics[dLambda_index_mechanics] = dLambda
                if "lmbda" in ep_model['parameter']:
                    p_ep[lmbda_index_ep] = lmbda_ti
                if "dLambda" in ep_model['parameter']:
                    p_ep[dLambda_index_ep] = dLambda

            if run_full_model:
                # Forward step for the full model
                y[:] = fgr(y, ti, dt, p)
                monitor = mon(ti, y, p)
                update_tracked_values(
                    index=i, monitor=monitor, y=y, p=p, tracked_values=track_values, model="full"
                )

            timing_ep_start = time.perf_counter()
            # Forward step for the EP model
            y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, *ep_missing_args)
            monitor_ep = mon_ep(ti, y_ep, p_ep, *ep_missing_args)

            update_tracked_values(
                index=i, monitor=monitor_ep, y=y_ep, p=p_ep, tracked_values=track_values, model="ep"
            )

            timing_ep_end = time.perf_counter()
            timings_ep_steps.append(timing_ep_end - timing_ep_start)

            # Update missing values for the mechanics model
            # this function just outputs the value of the missing values
            # straight from y_ep (does not calculate anything)
            mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, *ep_missing_args)

            if i % N != 0:
                count += 1
                # Lambda still needs to be updated:
                lmbda_ti = twitch(ti + dt)
                p[lmbda_index] = lmbda_ti
                p_mechanics[lmbda_index_mechanics] = lmbda_ti
                dLambda = (lmbda_ti - prev_lmbda) / dt
                p[dLambda_index] = dLambda
                p_mechanics[dLambda_index_mechanics] = dLambda
                prev_lmbda = lmbda_ti
                
                if "lmbda" in ep_model['parameter']:
                    p_ep[lmbda_index_ep] = lmbda_ti
                if "dLambda" in ep_model['parameter']:
                    p_ep[dLambda_index_ep] = dLambda
                
                timings_solveloop.append(time.perf_counter() - timing_loopstart)
                continue

            # Store the index of the time step where we performed a step
            inds.append(i)

            timing_mech_start = time.perf_counter()

            # For consistency with other models:
            y_mechanics[:] = fgr_mechanics(
                y_mechanics, ti, count * dt, p_mechanics, prev_mechanics_missing_values
            )

            count = 1
            monitor_mechanics = mon_mechanics(
                ti,
                y_mechanics,
                p_mechanics,
                mechanics_missing_values,
            )
            update_tracked_values(
                index=i,
                monitor=monitor_mechanics,
                y=y_mechanics,
                p=p_mechanics,
                tracked_values=track_values,
                model="mechanics",
            )

            timing_mech_end = time.perf_counter()
            timings_mech_steps.append(timing_mech_end - timing_mech_start)

            # Update lambda
            # Should be done after all calculations except ep_missing,
            # which is used for next ep step
            lmbda_ti = twitch(ti + dt)
            p[lmbda_index] = lmbda_ti
            p_mechanics[lmbda_index_mechanics] = lmbda_ti
            dLambda = (lmbda_ti - prev_lmbda) / dt
            p[dLambda_index] = dLambda
            p_mechanics[dLambda_index_mechanics] = dLambda
        
            if "lmbda" in ep_model['parameter']:
                p_ep[lmbda_index_ep] = lmbda_ti
            if "dLambda" in ep_model['parameter']:
                p_ep[dLambda_index_ep] = dLambda
            
            prev_lmbda = lmbda_ti

    

            # Update missing values for the EP model # J_TRPN for cai split
            if ep_missing_args:
                ep_missing_values[:] = mv_mechanics(
                    t, y_mechanics, p_mechanics, mechanics_missing_values
                )

            prev_mechanics_missing_values[:] = mechanics_missing_values

            timings_solveloop.append(time.perf_counter() - timing_loopstart)

        # Plot the results
        perc = 100 * len(inds) / len(t)
        print(f"Solved on {perc}% of the time steps")
        inds = np.array(inds)
        timing_total = time.perf_counter() - timing_init

        (outdir / f"split_cai_N{N}.json").write_text(
            json.dumps(
                {
                    "init_time": timing_init,
                    "loop_times": timings_solveloop,
                    "ep_step_times": timings_ep_steps,
                    "mech_step_times": timings_mech_steps,
                    "total_time": timing_total,
                }
            )
        )

        if save_traces:
            mech_header = np.array(
                [k for k, v in track_values.items() if v.mechanics_type != "none"]
            )
            mech_values = np.array(
                [v.mechanics_value for v in track_values.values() if v.mechanics_type != "none"]
            )
            np.savetxt(outdir / f"mech_N{N}.txt", mech_values.T, header=" ".join(mech_header))

            ep_header = np.array([k for k, v in track_values.items() if v.ep_type != "none"])
            ep_values = np.array([v.ep_value for v in track_values.values() if v.ep_type != "none"])
            np.savetxt(outdir / f"ep_N{N}.txt", ep_values.T, header=" ".join(ep_header))

            if run_full_model:
                full_header = np.array(
                    [k for k, v in track_values.items() if v.full_type != "none"]
                )
                full_values = np.array(
                    [v.full_value for v in track_values.values() if v.full_type != "none"]
                )
                np.savetxt(outdir / f"full_N{N}.txt", full_values.T, header=" ".join(full_header))

            import matplotlib.pyplot as plt

            for name, values in track_values.items():
                fig, ax = plt.subplots()
                if values.ep_type != "none":
                    ax.plot(t, values.ep_value, label="EP", alpha=0.5)
                if values.mechanics_type != "none":
                    ax.plot(t, values.mechanics_value, label="Mechanics", alpha=0.5)

                if run_full_model and values.full_type != "none":
                    ax.plot(t, values.full_value, label="Full", alpha=0.5)
                ax.set_title(name)
                ax.legend()
                ax.grid()
                fig.savefig(outdir / f"{name}_N{N}.png")
                plt.close(fig)
