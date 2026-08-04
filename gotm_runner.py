import os
import subprocess
import shutil

from utility import make_t_profile, edit_yaml, read_yaml_setting, write_run_settings


def run_gotm_experiments(
        root_dir: str,
        source_dir_name: str,
        training_set_name: str,
        case_dict: dict,
        depth: int = 400,
        vertical_spacing: float = 1.0
    ) -> list[str]:

    source_file = os.path.join(root_dir, source_dir_name, "gotm.yaml")
    dataset_dir = os.path.join(root_dir, f"{training_set_name}_training_dataset")

    nlev = int(read_yaml_setting(file=source_file, key="nlev"))
    model_depth = float(read_yaml_setting(file=source_file, key="depth"))

    write_run_settings(
        dataset_dir=dataset_dir,
        settings={
            "grid": "constant",
            "depth": model_depth,
            "nlev": nlev,
            "dz": model_depth / nlev,
            "dt": float(read_yaml_setting(file=source_file, key="time_step")),
            "model_dt": float(read_yaml_setting(file=source_file, key="dt")),
            "output_time_unit": read_yaml_setting(file=source_file, key="time_unit"),
            "output_time_method": read_yaml_setting(file=source_file, key="time_method"),
            "start": read_yaml_setting(file=source_file, key="start"),
            "stop": read_yaml_setting(file=source_file, key="stop"),
            "t_profile_depth": depth,
            "t_profile_spacing": vertical_spacing
        }
    )

    failed = []

    for case_ in case_dict.items():

        case_name, case_specs = case_
        case_dir = os.path.join(dataset_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        shutil.copy(source_file, case_dir)
        case_file = os.path.join(case_dir, "gotm.yaml")

        t_profile_pattern = r'^(?P<indent>\s*)file:\s+t_prof.dat'
        t_profile_file = make_t_profile(
            case_dir=case_dir,
            depth=depth,
            mld=1,
            temp_grad=case_specs["temp_grad"],
            vertical_spacing=vertical_spacing
        )

        output_pattern = r'^(?P<indent>\s*)output_filename'
        output_file = os.path.join(case_dir, f"output")

        tau_pattern = r'^(?P<indent>\s*)constant_value:\s+tx'
        heat_flux_pattern = r'^(?P<indent>\s*)constant_value:\s+heat_flux'
        latitude_pattern = r'^(?P<indent>\s*)latitude'

        patterns = [t_profile_pattern, output_pattern, tau_pattern, heat_flux_pattern, latitude_pattern]
        new_strings = [f"file: {t_profile_file}", f"{output_file}:", 
                       f"constant_value: {case_specs['tx']}", f"constant_value: {case_specs['heat_flux']}", f"latitude: {case_specs['lat']}"]

        for pattern, string in zip(patterns, new_strings):
            edit_yaml(
                file=case_file,
                pattern=pattern,
                new_string=string
            )
        
        # cwd rather than os.chdir, which left the process inside the last case directory
        result = subprocess.run(["gotm"], cwd=case_dir)

        if result.returncode != 0:
            failed.append(case_name)

    # a failed case leaves no output.nc, which otherwise surfaces much later as a confusing
    # FileNotFoundError from the feature retreiver
    if failed:
        print(f"{len(failed)} of {len(case_dict)} cases failed: {', '.join(failed)}")

    return failed
