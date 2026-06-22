import os
import shutil

from utility import make_t_profile, edit_yaml


def run_gotm_experiments(
        root_dir: str, 
        source_dir_name: str,
        training_set_name: str,
        case_dict: dict,
        depth: int = 400,
        vertical_spacing: float = 1.0
    ) -> None:

    source_file = os.path.join(root_dir, source_dir_name, "gotm.yaml")

    for case_ in case_dict.items():

        case_name, case_specs = case_
        case_dir = os.path.join(root_dir, f"{training_set_name}_training_dataset", case_name)
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
        
        os.chdir(case_dir)  # TODO: more sophisticated handling of failed runs? maybe not necessary 
        os.system('gotm')
