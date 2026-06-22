from gotm_runner import run_gotm_experiments
# from gotm_runner_f_u_star_grid import run_gotm_experiments
import json
from itertools import product

import numpy as np


training_set_name = "ePBL_paper_2283_corrected_dz0.1"
gotm_case_dict = {}
root_dir = "/your/root/dir/dataset_maker_for_gotm"

temperature_gradients = [0.001, 0.01, 0.02, 0.04]
wind_stresses = [0.01, 0.05, 0.1, 0.5, 1.0]
latitudes = np.arange(10., 100., 10.).tolist()
heat_fluxes = np.arange(-100,125,25).tolist()

for i, combo in enumerate(product(temperature_gradients, wind_stresses, latitudes, heat_fluxes)):
    temp_grad, tx, lat, hf = combo
    case_name = f"case_{i+1}"
    gotm_case_dict[case_name] = {"temp_grad": temp_grad, "tx": tx, "lat": lat, "heat_flux": hf}

with open(f"{root_dir}/{training_set_name}_training_set_cases.json", "w+") as file:
    json.dump(gotm_case_dict, file)

# with open(f"{root_dir}/{training_set_name}_training_set_cases.json", "r") as file:
#     gotm_case_dict = json.load(file)

# for i in range(1,35):
#     gotm_case_dict.pop(f"case_{i}")

run_gotm_experiments(
    root_dir=root_dir,
    source_dir_name="source_yaml",
    training_set_name=training_set_name,
    case_dict=gotm_case_dict
)