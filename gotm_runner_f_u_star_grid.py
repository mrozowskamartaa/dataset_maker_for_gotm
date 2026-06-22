import os
import shutil

from datetime import datetime, timedelta
from utility import calculate_T, make_t_profile, edit_yaml, compute_u_star, calculate_f


# The difference between this script and gotm_runner.py is that the vertical and temporal 
# resolutions of the runs are adjusted such that the number of points is relative to Ekman depth
# and the inertial period rather than a fixed max depth and max time for each case.


def get_time(
        T: float,
        n_inertial_periods: int = 20,
        points_per_inertial_period: int = 24,
        start_date: datetime = datetime(2011, 4, 1)
):
    time_step = int(T / points_per_inertial_period)
    total_time_days = int(n_inertial_periods * T / 24 / 60 / 60 + 1)
    stop_date = start_date + timedelta(days=total_time_days)
    return stop_date.strftime("%Y-%m-%d"), time_step


def run_gotm_experiments(
        root_dir: str, 
        source_dir_name: str,
        training_set_name: str,
        case_dict: dict,
        nlev: int = 10000
    ) -> None:

    source_file = os.path.join(root_dir, source_dir_name, "gotm.yaml")

    for case_ in case_dict.items():

        case_name, case_specs = case_
        case_dir = os.path.join(root_dir, f"{training_set_name}_training_dataset", case_name)
        os.makedirs(case_dir, exist_ok=True)
        shutil.copy(source_file, case_dir)
        case_file = os.path.join(case_dir, "gotm.yaml")

        f = calculate_f(latitude=case_specs['lat'])
        T = calculate_T(latitude=case_specs['lat'])
        u_star = compute_u_star(tau=case_specs['tx'])

        max_depth = int(u_star / f)
        stop_date, time_step = get_time(T=T)

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
        
        os.chdir(case_dir)  # TODO: more sophisticated handling of failed runs? maybe not necessary 
        os.system('gotm')
