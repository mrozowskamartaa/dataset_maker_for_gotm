from typing import Callable, Literal, Optional
import os

import xarray as xr
import numpy as np

from utility import (
    BL_DEFINITIONS, compute_ekman_layer_thickness, compute_ekman_spiral,
    calculate_f, compute_bl_depths, compute_u_star, interp_to_depth
)


class FeatureRetreiver:
    def __init__(
            self,
            training_set_dir: str,
            case_dict: dict,
            grid: Literal["constant", "f_u_star"],
            dz: Optional[int] = None,
            dt: Optional[int] = None,
            n_inertial_periods: Optional[int] = None,
            n_points_per_period: Optional[int] = None,
            bl_methods: Optional[list[str]] = None,
            min_bl_levels: int = 0
    ) -> None:

        self.grid = grid

        if self.grid == "constant":
            assert dz is not None, "dz must be provided for constant grid"
            assert dt is not None, "dt must be provided for constant grid"
        elif self.grid == "f_u_star":
            assert n_inertial_periods is not None, "n_inertial_periods must be provided for f_u_star grid"
            assert n_points_per_period is not None, "n_points_per_period must be provided for f_u_star grid"

        self.dz = dz
        self.dt = dt
        self.n_inertial_periods = n_inertial_periods
        self.n_points_per_period = n_points_per_period

        self.bl_methods = list(BL_DEFINITIONS.keys()) if bl_methods is None else bl_methods
        self.min_bl_levels = min_bl_levels

        self.alpha = -0.2
        self.grav = 9.81
        self.rho0 = 1027
        self.omega = 2 * np.pi / 24 / 60 / 60

        self.training_set_dir = training_set_dir
        self.case_dict = case_dict
        self.case_names = [key for key in case_dict.keys()]

        first_case = self.case_names[0]
        self.first_case_output = self.get_output(first_case)

        if self.grid == "constant":
            self.time = self.first_case_output['time'].values
        elif self.grid == "f_u_star":
            n_points = self.n_inertial_periods * self.n_points_per_period
            self.time = np.linspace(0, self.n_inertial_periods, n_points)


    def get_output(
            self,
            case: str
    ) -> xr.Dataset:
        output_file = os.path.join(self.training_set_dir, case, "output.nc")
        return xr.open_dataset(output_file).isel(lat=0, lon=0)


    def get_dz(
            self,
            output: xr.Dataset
    ) -> float:
        return output.z[0,1].values - output.z[0,0].values


    def get_dt(
            self,
            output: xr.Dataset
    ) -> float:
        delta = output.time[1].values - output.time[0].values
        return delta.total_seconds()


    def compute_storage_term(
            self,
            output: xr.Dataset
    ) -> np.ndarray:
        if self.grid == "f_u_star":
            dz, dt = self.get_dz(output=output), self.get_dt(output=output)
        elif self.grid == "constant":
            dz, dt = self.dz, self.dt
        return np.pad(((output['tke'].diff(dim="time") / dt).sum(dim="zi")*dz).values, pad_width=(0,1), mode="edge")
    

    def compute_buoyancy_mixing_term(
            self,
            output: xr.Dataset
    ) -> np.ndarray:
        if self.grid == "f_u_star":
            dz = self.get_dz(output=output)
        elif self.grid == "constant":
            dz = self.dz
        return ((output['nuh']*output['NN'].where(output['NN'] > 0)).sum(dim="zi")*dz).values


    def make_coords_dict(
            self,
            coords: list[str]
    ) -> dict:
        
        coords_dict = {}
        
        for coord in coords:
            if 'z' in coord:
                coords_dict[coord] = self.first_case_output[coord].isel(time=0).values
            else:
                coords_dict[coord] = self.first_case_output[coord].values

        coords_dict['case'] = self.case_names

        return coords_dict


    def make_coords_dict_f_u_star(
            self,
            coords: list[str]
    ) -> dict:
        
        coords_dict = {}
        
        for coord in coords:
            if 'z' in coord:
                nz = self.first_case_output[coord].isel(time=0).values
                coords_dict[coord] = np.arange(nz)
            elif 'time' in coord:
                coords_dict[coord] = self.time

        coords_dict['case'] = self.case_names

        return coords_dict
    

    def make_m_star_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }

        # TODO: It would have been better to make case dict have int as keys and then 
        # construct the case names based on those.
        # Consider this edit in future generation of training datasets.
        
        m_star = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            wb = -output.G.values
            wb[wb < 0] = 0
            u_star = compute_u_star(
                tau=self.case_dict[case]['tx'], 
                rho=self.rho0
            )

            if self.grid == "f_u_star":
                dz = self.get_dz(output=output)
                m_star[i] = (np.sum(wb, axis=1) * dz / u_star ** 3)[:len(self.time)]
            elif self.grid == "constant":
                dz = self.dz
                m_star[i] = np.sum(wb, axis=1) * dz / u_star ** 3

        if self.grid == "f_u_star":
            m_star = m_star[:,:len(self.time)]
        
        data_vars = {"m_star": xr.DataArray(
            m_star,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
    

    def make_M_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }

        M = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            wb = -output.G.values
            wb[wb < 0] = 0

            if self.grid == "f_u_star":
                dz = self.get_dz(output=output)
                M[i] = (np.sum(wb, axis=1) * dz)[:len(self.time)]
            elif self.grid == "constant":
                dz = self.dz
                M[i] = np.sum(wb, axis=1) * dz

        data_vars = {"M": xr.DataArray(
            M,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def make_wb_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }

        M = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            wb = -output.G.values
            wb[wb > 0] = 0

            if self.grid == "f_u_star":
                dz = self.get_dz(output=output)
                M[i] = (np.sum(wb, axis=1) * dz)[:len(self.time)]
            elif self.grid == "constant":
                dz = self.dz
                M[i] = np.sum(wb, axis=1) * dz

        data_vars = {"wb": xr.DataArray(
            M,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def align_time(
            self,
            array: np.ndarray
    ) -> np.ndarray:
        return array[:len(self.time)]


    def bl_attrs(
            self,
            bl_methods: list[str]
    ) -> dict:
        return {
            "sigma_convention": "depth below surface / boundary layer depth; 0 at the surface, 1 at the boundary layer base",
            "bl_definitions": "; ".join(
                f"{method}: {BL_DEFINITIONS[method]['variable']} < {BL_DEFINITIONS[method]['threshold']:g}"
                for method in bl_methods
            ),
            "min_bl_levels": self.min_bl_levels
        }


    def sample_at_sigma(
            self,
            output: xr.Dataset,
            variable: str,
            sigma: np.ndarray,
            bl_methods: list[str]
    ) -> np.ndarray:
        """Sample `variable` at each sigma, under each boundary layer definition.

        Returns (bl_method, time, sigma). Each sigma is turned into a physical depth against
        that definition's boundary layer depth and the variable is interpolated onto it along
        its own vertical coordinate, so nothing is re-indexed across the staggered grids and
        an unresolved boundary layer propagates as NaN rather than as a plausible number.
        """
        sigma = np.asarray(sigma, dtype=float)
        bl_depths = compute_bl_depths(
            output=output, methods=bl_methods, min_levels=self.min_bl_levels
        )
        return np.stack([
            interp_to_depth(
                output=output,
                variable=variable,
                depths=bl_depths[method][:, np.newaxis] * sigma[np.newaxis, :]
            )
            for method in bl_methods
        ])


    def make_bl_depth_dataset(
            self,
            bl_methods: Optional[list[str]] = None
    ) -> xr.Dataset:

        bl_methods = self.bl_methods if bl_methods is None else bl_methods

        coords = {
            'case': self.case_names,
            'time': self.time,
            'bl_method': bl_methods
        }

        array = np.empty((len(self.case_names), len(self.time), len(bl_methods)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            bl_depths = compute_bl_depths(
                output=output, methods=bl_methods, min_levels=self.min_bl_levels
            )
            array[i] = self.align_time(
                np.stack([bl_depths[method] for method in bl_methods], axis=-1)
            )

        data_vars = {"bl": xr.DataArray(
            array,
            dims=['case', 'time', 'bl_method'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.bl_attrs(bl_methods=bl_methods)
        )


    def make_sigma_profile_dataset(
            self,
            variable: str,
            sigma_grid: list,
            bl_methods: Optional[list[str]] = None
    ) -> xr.Dataset:

        bl_methods = self.bl_methods if bl_methods is None else bl_methods
        sigma_grid = np.asarray(sigma_grid, dtype=float)

        coords = {
            'case': self.case_names,
            'time': self.time,
            'sigma_depth': sigma_grid,
            'bl_method': bl_methods
        }

        array = np.empty(
            (len(self.case_names), len(self.time), len(sigma_grid), len(bl_methods))
        )

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            sampled = self.sample_at_sigma(
                output=output,
                variable=variable,
                sigma=sigma_grid,
                bl_methods=bl_methods
            )
            array[i] = self.align_time(np.moveaxis(sampled, 0, -1))

        data_vars = {f"{variable}": xr.DataArray(
            array,
            dims=['case', 'time', 'sigma_depth', 'bl_method'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.bl_attrs(bl_methods=bl_methods)
        )
    
    
    def make_var_at_bl_dataset(
        self,
        variable: str,
        sigma: float = 0.9,
        log: bool = False,
        bl_methods: Optional[list[str]] = None
    ) -> xr.Dataset:

        bl_methods = self.bl_methods if bl_methods is None else bl_methods

        coords = {
            'case': self.case_names,
            'time': self.time,
            'bl_method': bl_methods
        }

        array = np.empty((len(self.case_names), len(self.time), len(bl_methods)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            sampled = self.sample_at_sigma(
                output=output,
                variable=variable,
                sigma=[sigma],
                bl_methods=bl_methods
            )[:, :, 0]
            if log:
                sampled = np.log10(np.where(sampled > 0, sampled, np.nan))
            array[i] = self.align_time(sampled.T)

        data_vars = {f"{variable}_bl": xr.DataArray(
            array,
            dims=['case', 'time', 'bl_method'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.bl_attrs(bl_methods=bl_methods)
        )


    def make_var_across_bl_dataset(
        self,
        variable: str,
        sigma_above: float = 0.9,
        sigma_below: float = 1.1,
        bl_methods: Optional[list[str]] = None
    ) -> xr.Dataset:

        bl_methods = self.bl_methods if bl_methods is None else bl_methods

        coords = {
            'case': self.case_names,
            'time': self.time,
            'bl_method': bl_methods
        }

        array = np.empty((len(self.case_names), len(self.time), len(bl_methods)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            sampled = self.sample_at_sigma(
                output=output,
                variable=variable,
                sigma=[sigma_above, sigma_below],
                bl_methods=bl_methods
            )
            array[i] = self.align_time(sampled.mean(axis=-1).T)

        data_vars = {f"{variable}_bl_mean": xr.DataArray(
            array,
            dims=['case', 'time', 'bl_method'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.bl_attrs(bl_methods=bl_methods)
        )
    

    def make_buoyancy_mixing_term_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }
        
        buoyancy_mixing_term = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            if self.grid == "f_u_star":
                buoyancy_mixing_term[i] = self.compute_buoyancy_mixing_term(output=output)[:len(self.time)]
            elif self.grid == "constant":
                buoyancy_mixing_term[i] = self.compute_buoyancy_mixing_term(output=output)
        
        data_vars = {"buoyancy_mixing_term": xr.DataArray(
            buoyancy_mixing_term,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def make_storage_term_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }
        
        storage_term = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            if self.grid == "f_u_star":
                storage_term[i] = self.compute_storage_term(output=output)[:len(self.time)]
            elif self.grid == "constant":
                storage_term[i] = self.compute_storage_term(output=output)
        
        data_vars = {"storage_term": xr.DataArray(
            storage_term,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def make_ekman_filtered_uv_dataset(self) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }

        u_i = np.empty((len(self.case_names), len(self.time)))
        v_i = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            u_star = compute_u_star(
                tau=self.case_dict[case]['tx'], 
                rho=self.rho0
            )  # TODO: WTF is this tho
            tau = u_star ** 2 * self.rho0
            f = calculate_f(latitude=self.case_dict[case]['lat'])
            nu = 0.1
            delta = compute_ekman_layer_thickness(nu=nu, f=f)
            z = output['z'].values[0]
            u_ekman, v_ekman = compute_ekman_spiral(
                tau=tau, 
                rho=self.rho0, 
                z=z, 
                delta=delta, 
                f=f, 
                nu=nu
            )
            u_i[i] = (output.u.T.values - u_ekman[:, np.newaxis])[-1,:len(self.time)]
            v_i[i] = (output.v.T.values - v_ekman[:, np.newaxis])[-1,:len(self.time)]

        data_vars = {
            "u_i": xr.DataArray(
                u_i,
                dims=['case', 'time'],
                coords=coords
            ),
            "v_i": xr.DataArray(
                v_i,
                dims=['case', 'time'],
                coords=coords
            )
        }

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def make_dataset_from_processed_data(
            self,
            processing_method: Callable[[xr.Dataset, str], np.ndarray],
            processing_method_name: str,
            variable: str,
            coordinates: list[str]
    ) -> xr.Dataset:
        
        if self.grid == "f_u_star":
            coords = self.make_coords_dict_f_u_star(coords=coordinates)
        elif self.grid == "constant":
            coords = self.make_coords_dict(coords=coordinates)

        coordinates = ['case'] + coordinates

        test_array = processing_method(
            output=self.first_case_output,
            variable=variable
        )

        if "time" in coordinates and self.grid == "f_u_star":
            test_array = test_array[:len(self.time)]

        array = np.empty((len(self.case_names),) + test_array.shape)

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            if "time" in coordinates and self.grid == "f_u_star":
                array[i] = processing_method(
                output=output, 
                variable=variable
            )[:len(self.time)]
            else:                
                array[i] = processing_method(
                output=output, 
                variable=variable
            )
            
        
        data_vars = {f"{processing_method_name}" : xr.DataArray(
            array,
            dims=coordinates,
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )


    def make_dataset_from_raw_data(
            self,
            variable: str,
            coordinates: list[str]
    ) -> xr.Dataset:

        if self.grid == "f_u_star":
            coords = self.make_coords_dict_f_u_star(coords=coordinates)
        elif self.grid == "constant":
            coords = self.make_coords_dict(coords=coordinates)

        coordinates = ['case'] + coordinates

        if self.grid == "f_u_star" and "time" in coordinates:
            shape = self.first_case_output[variable][:len(self.time)].shape
        else:
            shape = self.first_case_output[variable].shape

        array = np.empty((len(self.case_names),) + shape)

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            if "time" in coordinates and self.grid == "f_u_star":
                array[i] = output[variable].values[:len(self.time)]
            else:                
                array[i] = output[variable].values
        
        data_vars = {variable : xr.DataArray(
            array,
            dims=coordinates,
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
