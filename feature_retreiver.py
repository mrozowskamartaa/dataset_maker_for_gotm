from typing import Callable, Literal, Optional
import os

import xarray as xr
import numpy as np

from scipy.interpolate import make_smoothing_spline

from utility import (
    compute_ekman_layer_thickness, compute_ekman_spiral,
    calculate_f, compute_u_star, sample_from_bl_grid
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
            n_points_per_period: Optional[int] = None
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


    def make_sigma_profile_dataset(
            self,
            variable: str,
            sigma_grid: list
    ) -> xr.Dataset:
        
        coords = {
            'case': self.case_names,
            'time': self.time,
            'sigma_depth': sigma_grid
        }

        array = np.empty((len(self.case_names), len(self.time), len(sigma_grid)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            array[i] = sample_from_bl_grid(
                output=output,
                variable=variable,
                depths_to_sample_at=sigma_grid
            ).T

        data_vars = {f"{variable}": xr.DataArray(
            array,
            dims=['case', 'time', 'sigma_depth'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
    
    
    def make_var_at_bl_dataset(
        self,
        variable: str,
        index_above_bl: int = 2,
        smoothed: bool = False,
        log: bool = False,
        z_max: int = 400
    ) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }
        
        array = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            t, z = output[variable].shape
            z_grid = np.arange(z)[::-1]
            if z == z_max:
                z_mask = np.where(output['nuh'].values[:,:-1] > 1e-6, z_grid, np.nan)
            elif z == int(z_max + 1):
                z_mask = np.where(output['nuh'].values > 1e-6, z_grid, np.nan)
            z_indices = np.nanmax(z_mask, axis=1)
            z_indices = np.where(np.isnan(z_indices), 1, z_indices).astype(int)
            var = np.log10(output[variable].values[np.arange(t),-z_indices+index_above_bl]) if log else output[variable].values[np.arange(t),-z_indices+index_above_bl]
            if smoothed:
                ran = np.arange(t)
                smoothing = make_smoothing_spline(np.arange(len(var[1:])), var[1:], lam=10)
                array[i] = smoothing(ran) if self.grid == "constant" else smoothing(ran[:len(self.time)])
            else:
                array[i] = var if self.grid == "constant" else var[:len(self.time)]
        
        data_vars = {f"{variable}_bl": xr.DataArray(
            array,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )

    
    def make_var_across_bl_dataset(
        self,
        variable: str,
        index_above_bl: int = 2,
        index_below_bl: int = 2,
        z_max: int = 400
    ) -> xr.Dataset:

        coords = {
            'case': self.case_names,
            'time': self.time
        }
        
        array = np.empty((len(self.case_names), len(self.time)))

        for i, case in enumerate(self.case_dict.keys()):
            output = self.get_output(case)
            t, z = output[variable].shape
            z_grid = np.arange(z)[::-1]
            if z == z_max:
                z_mask = np.where(output['nuh'].values[:,:-1] > 1e-6, z_grid, np.nan)
            elif z == int(z_max + 1):
                z_mask = np.where(output['nuh'].values > 1e-6, z_grid, np.nan)
            z_indices = np.nanmax(z_mask, axis=1)
            z_indices = np.where(np.isnan(z_indices), 1, z_indices).astype(int)
            val_1 = output[variable].values[np.arange(t),-z_indices+index_above_bl]
            val_2 = output[variable].values[np.arange(t),-z_indices-index_below_bl]
            if self.grid == "constant":
                array[i] = (val_1 + val_2) / (index_above_bl + index_below_bl)
            elif self.grid == "f_u_star":
                array[i] = ((val_1 + val_2) / (index_above_bl + index_below_bl))[:len(self.time)]
        
        data_vars = {f"{variable}_bl_mean": xr.DataArray(
            array,
            dims=['case', 'time'],
            coords=coords
        )}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords
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
