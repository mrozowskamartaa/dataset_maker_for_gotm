import os
import subprocess
import shutil

from datetime import datetime, timedelta
from utility import (
    calculate_T, make_t_profile, edit_yaml, compute_u_star, calculate_f,
    read_yaml_setting, write_run_settings, compute_N0, compute_nonrot_mixed_layer_depth,
    compute_nonrot_end_time, SECONDS_PER_DAY
)


# The difference between this script and gotm_runner.py is that the vertical and temporal 
# resolutions of the runs are adjusted such that the number of points is relative to Ekman depth
# and the inertial period rather than a fixed max depth and max time for each case.
#
# Cases at f = 0 have neither, so they are scaled by the non-rotating deepening law
# D(t) = 1.2 u_star (t / N0)^(1/2) instead: the run ends at t_end, set through
# nonrot_end_depth_ratio = D(t_end) / D(1 day) or directly through nonrot_end_time_days, and
# the domain is 2 D(t_end) deep. They keep the same number of output samples as the rotating
# cases, n_inertial_periods * points_per_inertial_period, spread evenly over t_end - so their
# time scale (the stand-in for T) is t_end / n_inertial_periods.


def get_time(
        time_scale: float,
        n_inertial_periods: int = 20,
        points_per_inertial_period: int = 24,
        start_date: datetime = datetime(2011, 4, 1)
):
    time_step = int(time_scale / points_per_inertial_period)
    total_time_days = int(n_inertial_periods * time_scale / SECONDS_PER_DAY + 1)
    stop_date = start_date + timedelta(days=total_time_days)
    return stop_date.strftime("%Y-%m-%d"), time_step


def get_case_scales(
        case_specs: dict,
        n_inertial_periods: int,
        nonrot_end_depth_ratio: float = None,
        nonrot_end_time_days: float = None
) -> dict:
    """Depth scale [m] and time scale [s] of one case, which set its domain depth and run length.

    Rotating: the Ekman depth u_star / f and the inertial period T. Non-rotating: twice the
    deepening-law depth at the end of the run, 2 D(t_end), and t_end / n_inertial_periods.
    """
    f = calculate_f(latitude=case_specs['lat'])
    u_star = compute_u_star(tau=case_specs['tx'])

    if f != 0:
        return {
            "rotating": True,
            "depth_scale": int(u_star / f),
            "time_scale": calculate_T(latitude=case_specs['lat'])
        }

    t_end = compute_nonrot_end_time(
        end_depth_ratio=nonrot_end_depth_ratio,
        end_time_days=nonrot_end_time_days
    )
    N0 = compute_N0(temp_grad=case_specs['temp_grad'])
    end_depth = compute_nonrot_mixed_layer_depth(u_star=u_star, N0=N0, t=t_end)

    return {
        "rotating": False,
        "depth_scale": int(2 * end_depth),
        "time_scale": t_end / n_inertial_periods,
        "N0": N0,
        "day_one_depth": compute_nonrot_mixed_layer_depth(u_star=u_star, N0=N0, t=SECONDS_PER_DAY),
        "end_depth": end_depth,
        "end_time_days": t_end / SECONDS_PER_DAY
    }


def run_gotm_experiments(
        root_dir: str,
        source_dir_name: str,
        training_set_name: str,
        case_dict: dict,
        nlev: int = 10000,
        n_inertial_periods: int = 20,
        points_per_inertial_period: int = 24,
        nonrot_end_depth_ratio: float = None,
        nonrot_end_time_days: float = None
    ) -> list[str]:
    """Run every case on the f_u_star grid.

    For f = 0 cases set exactly one of
        nonrot_end_depth_ratio: D(t_end) / D(1 day), how far the layer deepens relative to
            its depth after one day (2 -> D(t_end) = 2 D(1 day), run length 4 days)
        nonrot_end_time_days: run length t_end in days
    Neither is used, and both may be left unset, when every case rotates.
    """

    source_file = os.path.join(root_dir, source_dir_name, "gotm.yaml")
    dataset_dir = os.path.join(root_dir, f"{training_set_name}_training_dataset")

    # all scales up front, so a bad setting fails before any GOTM run rather than partway
    # through the set
    case_scales = {
        case_name: get_case_scales(
            case_specs=case_specs,
            n_inertial_periods=n_inertial_periods,
            nonrot_end_depth_ratio=nonrot_end_depth_ratio,
            nonrot_end_time_days=nonrot_end_time_days
        )
        for case_name, case_specs in case_dict.items()
    }

    for case_name, scales in case_scales.items():
        if scales["depth_scale"] < 1:
            raise ValueError(f"{case_name}: domain depth {scales['depth_scale']} m is under 1 m")
        if int(scales["time_scale"] / points_per_inertial_period) < 1:
            raise ValueError(f"{case_name}: output interval rounds to 0 s; lengthen the run")

    write_run_settings(
        dataset_dir=dataset_dir,
        settings={
            "grid": "f_u_star",
            "nlev": nlev,
            "n_inertial_periods": n_inertial_periods,
            "points_per_inertial_period": points_per_inertial_period,
            "nonrot_end_depth_ratio": nonrot_end_depth_ratio,
            "nonrot_end_time_days": nonrot_end_time_days,
            "model_dt": float(read_yaml_setting(file=source_file, key="dt")),
            "output_time_unit": read_yaml_setting(file=source_file, key="time_unit"),
            "output_time_method": read_yaml_setting(file=source_file, key="time_method"),
            # depth is u_star / f (2 D(t_end) at f = 0) and the output interval is
            # time_scale / points_per_inertial_period, so both are per case; see case_scales
            "depth": "per case",
            "dz": "per case",
            "dt": "per case",
            "case_scales": case_scales
        }
    )

    failed = []

    for case_ in case_dict.items():

        case_name, case_specs = case_
        case_dir = os.path.join(dataset_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        shutil.copy(source_file, case_dir)
        case_file = os.path.join(case_dir, "gotm.yaml")

        max_depth = case_scales[case_name]["depth_scale"]
        stop_date, time_step = get_time(
            time_scale=case_scales[case_name]["time_scale"],
            n_inertial_periods=n_inertial_periods,
            points_per_inertial_period=points_per_inertial_period
        )

        t_profile_pattern = r'^(?P<indent>\s*)file:\s+t_prof.dat'
        t_profile_file = make_t_profile(
            case_dir=case_dir,
            depth=max_depth,
            mld=1,
            temp_grad=case_specs["temp_grad"],
            vertical_spacing=1.0
        )

        output_pattern = r'^(?P<indent>\s*)output_filename'
        output_file = os.path.join(case_dir, f"output")

        tau_pattern = r'^(?P<indent>\s*)constant_value:\s+tx'
        heat_flux_pattern = r'^(?P<indent>\s*)constant_value:\s+heat_flux'
        latitude_pattern = r'^(?P<indent>\s*)latitude'
        
        depth_pattern = r'^(?P<indent>\s*)depth:\s+\d+'
        nlev_pattern = r'^(?P<indent>\s*)nlev:\s+\d+'
        stop_pattern = r'^(?P<indent>\s*)stop:\s+\d+'
        time_step_pattern = r'^(?P<indent>\s*)time_step:\s+\d+'

        patterns = [
            t_profile_pattern, output_pattern, tau_pattern, heat_flux_pattern, 
            latitude_pattern, depth_pattern, nlev_pattern, stop_pattern, time_step_pattern
        ]

        new_strings = [
            f"file: {t_profile_file}", f"{output_file}:", f"constant_value: {case_specs['tx']}", 
            f"constant_value: {case_specs['heat_flux']}", f"latitude: {case_specs['lat']}",
            f"depth: {max_depth}", f"nlev: {nlev}", f"stop: {stop_date}", f"time_step: {time_step}"
        ]

        for pattern, string in zip(patterns, new_strings):
            edit_yaml(
                file=case_file,
                pattern=pattern,
                new_string=string
            )
        
        # an output.nc left by an earlier run would otherwise survive a failed rerun and be
        # read as this run's result
        output_nc = f"{output_file}.nc"
        if os.path.exists(output_nc):
            os.remove(output_nc)

        # cwd rather than os.chdir, which left the process inside the last case directory
        result = subprocess.run(["gotm"], cwd=case_dir)

        if result.returncode != 0 or not os.path.exists(output_nc):
            failed.append(case_name)

    # a failed case leaves no output.nc, which otherwise surfaces much later as a confusing
    # FileNotFoundError from the feature retreiver
    if failed:
        print(f"{len(failed)} of {len(case_dict)} cases failed: {', '.join(failed)}")

    return failed
