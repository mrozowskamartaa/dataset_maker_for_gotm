import re
import os
from datetime import datetime, timedelta


import numpy as np
import xarray as xr


### --- PHYSICS --- ###


def calculate_f(latitude: float) -> float:
    omega = 2 * np.pi / 24 / 60 / 60
    return 2 * omega * np.sin(latitude * np.pi / 180)


def calculate_T(latitude: float) -> float:
    return 2 * np.pi / calculate_f(latitude)


def calculate_B(
        heat_fluxes: np.ndarray,
        rho0: float = 1027,
        cp: float = 4e3,
        alpha: float = 2e-4,
        g: float = 9.81 
) -> np.ndarray:
    return g * alpha * heat_fluxes / (rho0 * cp)


def compute_u_star(
        tau: float,
        rho: float = 1027
) -> float:
    return (tau / rho) ** 0.5


def calculate_analytical_m_star_N(
        x: np.ndarray, 
        c_N1: float, 
        c_N2: float, 
        c_N3: float
) -> np.ndarray:
    return c_N1 * (1 - (1 + c_N2 * np.exp(-c_N3*x)) ** -1)


def calculate_analytical_m_star_Nb(
        x: np.ndarray, 
        c_Nb1: float, 
        c_Nb2: float
) -> np.ndarray:
    return c_Nb1 * np.exp(-c_Nb2 * x)


def compute_tke_bl(
        output: xr.Dataset,
        variable: str = "tke"
) -> np.ndarray:
    z_above = output.zi.where(output[variable] > 1e-9).min(dim="zi")
    z_below = output.zi.where(output[variable] < 1e-9).max(dim="zi")

    tke_above = output[variable].where(output[variable] > 1e-9).min(dim="zi")
    tke_below = output[variable].where(output[variable] < 1e-9).max(dim="zi")

    return -1*(-z_above+z_below)/(tke_above-tke_below)*(1e-9-tke_below) + z_below


def compute_ekman_layer_thickness(
        nu: float, 
        f: float
) -> float:
    return np.sqrt(2 * nu / np.abs(f))


def compute_ekman_spiral(
        tau: float, 
        rho: float, 
        z: np.ndarray, 
        delta: float, 
        f: float, 
        nu: float
) -> tuple[np.ndarray, np.ndarray]:
    u = tau / (rho * np.sqrt(f * nu)) * np.exp(z / delta) * np.cos(-z / delta + np.pi / 4)
    v = tau / (rho * np.sqrt(f * nu)) * np.exp(z / delta) * np.sin(-z / delta + np.pi / 4)
    return u, v


def compute_bl(
        output: xr.Dataset,
        variable: str = "eps"
) -> np.ndarray:
    bl_mask = np.where(output[variable].values > 1e-12, -output['zi'].values, np.nan)
    bl = np.nanmax(bl_mask, axis=1)
    return bl
    

def compute_bl_rh18(
        output: xr.Dataset,
        variable: str = "nuh"
) -> np.ndarray:
    bl_mask = np.where(output[variable].values > 1e-6, -output['zi'].values, np.nan)
    bl = np.nanmax(bl_mask, axis=1)
    surface_z = -output['z'].values[0,-1]
    bl = np.where(np.isnan(bl), surface_z, bl)
    return bl


### --- INTERPOLATED BOUNDARY LAYER DEPTH --- ###


# Each definition is a (variable, threshold) pair. The threshold must sit clear of the
# variable's floor in gotm.yaml, otherwise the crossing lands on the floor and no sub-grid
# interpolation is possible: eps_min is 1e-12, so the eps definition below is degenerate as
# it stands and wants raising to ~1e-11 before it can be compared against the others.
BL_DEFINITIONS = {
    "rh18": {"variable": "nuh", "threshold": 1e-6},
    "eps": {"variable": "eps", "threshold": 1e-12},
    "tke": {"variable": "tke", "threshold": 1e-9},
}


def vertical_dim(array: xr.DataArray) -> str:
    for dim in ("zi", "z"):
        if dim in array.dims:
            return dim
    raise ValueError(f"{array.name} has no vertical dimension: {array.dims}")


def find_bl_depth(
        output: xr.Dataset,
        variable: str = "nuh",
        threshold: float = 1e-6,
        min_levels: int = 0
) -> np.ndarray:
    """Boundary layer depth [m, positive down], interpolated to the sub-grid crossing.

    Scans downward from the surface and stops at the first level where `variable` drops
    below `threshold`, then interpolates the crossing between that level and the one above.
    Scanning from the surface rather than taking the deepest crossing keeps detached mixing
    at depth (a bottom log-layer, say) out of the surface boundary layer, and works for
    profiles that are not monotone in depth - nuh, which falls to zero at both the surface
    and the base, is not, which is why masked-extrema interpolation misbehaves on it.

    NaN where the surface level is itself below threshold, and where the layer spans fewer
    than `min_levels` cells. The full column depth where nothing crosses.
    """
    var = output[variable].values
    z = output[vertical_dim(output[variable])].values

    # GOTM writes profiles bottom-up, so flip to run the scan from the surface down
    var, z = var[:, ::-1], z[:, ::-1]

    below = var < threshold
    crosses = below.any(axis=1)
    k = np.argmax(below, axis=1)  # first sub-threshold level, 0 where nothing crosses

    rows = np.arange(var.shape[0])
    k_safe = np.clip(k, 1, var.shape[1] - 1)

    var_above, var_below = var[rows, k_safe - 1], var[rows, k_safe]
    z_above, z_below = z[rows, k_safe - 1], z[rows, k_safe]

    span = var_above - var_below
    weight = np.divide(
        var_above - threshold, span, out=np.zeros_like(span), where=span != 0
    )

    bl = z[:, 0] - (z_above + weight * (z_below - z_above))
    bl[~crosses] = (z[:, 0] - z[:, -1])[~crosses]
    bl[crosses & (k < max(min_levels, 1))] = np.nan

    return bl


def compute_bl_depths(
        output: xr.Dataset,
        methods: list[str] = None,
        min_levels: int = 0
) -> dict[str, np.ndarray]:
    methods = list(BL_DEFINITIONS.keys()) if methods is None else methods
    return {
        method: find_bl_depth(output=output, min_levels=min_levels, **BL_DEFINITIONS[method])
        for method in methods
    }


### --- MATH --- ###


def calculate_rolling_average(
        case: str,
        dataset: xr.DataArray,
        variable: str,
        case_dict: dict,
        dt: int = 600
) -> np.ndarray:
    data = dataset[variable].sel(case=case).values
    lat = case_dict[case]['lat']
    T = calculate_T(lat)
    window_size = int(T/dt)
    return np.convolve(
        np.pad(data, pad_width=int(window_size/2), mode="reflect"), np.ones(window_size)/window_size, mode='valid'
    )[:-1]


def calculate_mean_over_inertial_period(
        data: np.ndarray,
        latitudes: np.ndarray,
        dt: float = 50
) -> np.ndarray:
    m, _ = data.shape
    ten_days = 10 * 24 * 60 * 60  # HARD CODED BABEY; this is the issue with the long runs; should be accounted for
    averages, variances = np.empty(m), np.empty(m)
    for i in range(m):
        if latitudes[i] <= 5:
            averages[i], variances[i] = np.nanmean(data[i]), np.nanstd(data[i])
        else:
            inertial_period = calculate_T(latitude=latitudes[i])
            n_steps_per_period = int(inertial_period / dt)
            array = data[i, n_steps_per_period:]  # discard the first inertial period
            n_chunks = int(ten_days / inertial_period) - 1
            average, variance = np.empty(n_chunks + 1), np.empty(n_chunks + 1)
            for j in range(n_chunks):
                start, end = n_steps_per_period*j, n_steps_per_period*(j+1)
                average[j], variance[j] = np.nanmean(array[start:end]), np.nanstd(array[start:end]) ** 2
            average[n_chunks], variance[n_chunks] = np.nanmean(array[end:]), np.nanstd(array[end:]) ** 2
            averages[i], variances[i] = np.nanmean(average), np.nanmean(variance)
    return averages, variances


def last_profile(
        output: xr.Dataset,
        variable: str
) -> np.ndarray:
    return output[variable].isel(time=-1).values


### --- DATA PROCESSING --- ###


def interp_to_depth(
        output: xr.Dataset,
        variable: str,
        depths: np.ndarray
) -> np.ndarray:
    """Linearly interpolate `variable` to `depths` [m below the surface], one row per time.

    Interpolates along the variable's own vertical coordinate, so centred variables (u, v,
    temp) and interface variables (nuh, NN, Rig) can be sampled at the same physical depth
    without either being re-indexed onto the other's staggered grid. NaN outside the water
    column, and wherever `depths` is NaN.
    """
    array = output[variable]
    dim = vertical_dim(array)

    # GOTM writes profiles bottom-up, so flip to surface-first and measure downward
    z = output[dim].values[:, ::-1]
    values = array.transpose("time", dim).values[:, ::-1]

    if not np.allclose(z[0], z[-1]):
        raise ValueError("vertical grid varies in time; interpolation assumes it does not")

    grid = z[0, 0] - z[0]

    index = np.clip(np.searchsorted(grid, depths), 1, grid.size - 1)
    weight = (depths - grid[index - 1]) / (grid[index] - grid[index - 1])

    shallow = np.take_along_axis(values, index - 1, axis=1)
    deep = np.take_along_axis(values, index, axis=1)

    return np.where(
        (depths < grid[0]) | (depths > grid[-1]),
        np.nan,
        shallow + weight * (deep - shallow)
    )


def interp_within_bl(
        a: np.ndarray,
        nz_tot: int
) -> np.ndarray:
    """Stretch one time step's in-boundary-layer values onto a fixed sigma grid.

    `a` arrives bottom-up with NaN outside the boundary layer, so its valid values run from
    the layer base up to the surface. Reversing them puts the surface at index 0, i.e. at
    sigma = 0, and the layer base at sigma = 1. Stretching per time step is what makes sigma
    relative to the instantaneous boundary layer depth rather than to the deepest layer
    reached anywhere in the run.
    """
    mask = np.isnan(a)

    if mask.all():
        return np.zeros(nz_tot)

    valid = a[~mask][::-1]

    if valid.shape[0] == 1:
        return np.ones(nz_tot) * valid

    return np.interp(
        np.linspace(0, 1, nz_tot), np.linspace(0, 1, valid.shape[0]), valid
    )


def sample_from_bl_grid(
    output: xr.Dataset,
    variable: str,
    depths_to_sample_at: np.ndarray,
    bl_threshold: float = 1e-9,
    bl_variable: str = "tke"
) -> np.ndarray:
    masked_var = output[variable].where(output[bl_variable] > bl_threshold).T.values
    cleaned_var = masked_var[~np.isnan(masked_var).all(axis=1)]

    var = np.apply_along_axis(
        interp_within_bl, axis=0, arr=cleaned_var, nz_tot=cleaned_var.shape[0]
    )

    # ascending to match interp_within_bl: sigma 0 at the surface, 1 at the layer base
    sigma_grid = np.linspace(0, 1, var.shape[0])
    sigma = np.asarray(depths_to_sample_at, dtype=float)
    indices = np.argmin(np.abs(sigma_grid[:, np.newaxis] - sigma[np.newaxis, :]), axis=0)

    return var[indices, :]


def surface_value_z_grid(
        output: xr.Dataset,
        variable: str
) -> np.ndarray:
    return output[variable].isel(z=-1).values


def get_time_stamps_for_profiles(
    dt: int,
    latitude: float,
    time_intervals: list
):
    inertial_period = calculate_T(latitude)
    return [int(inertial_period*interval / dt) for interval in time_intervals]


### --- GOTM RUNNER UTILITY FUNCTIONS --- ###


def generate_time_range(
        start: datetime, 
        end: datetime, 
        delta: int
):
    current_date = start
    while current_date < end:
        yield current_date
        current_date += timedelta(days=delta)


def make_t_profile(
        case_dir: str,
        depth: int,
        mld: int,
        temp_grad: float = 0.04,
        mld_temp: float = 29.25,
        vertical_spacing: float = 1.0
) -> str:
    
    nz = int(depth / vertical_spacing)
    mld_index = int(mld / vertical_spacing)
    mid_nz = vertical_spacing / 2
    temp_grad_per_nz = temp_grad * vertical_spacing

    t_profile = np.empty(nz)
    t_profile[0:mld_index] = mld_temp
    t_profile[mld_index:] = mld_temp - temp_grad_per_nz*np.arange(1,nz-mld_index+1,1)

    z_start = - depth + mid_nz
    z_end = mid_nz
    z_levels = np.arange(z_start,z_end,vertical_spacing)[::-1]

    date_start = datetime(2011, 4, 1)
    date_end = datetime(2012, 5, 1)
    delta = 360
    date_list = list(generate_time_range(date_start, date_end, delta))

    filename = os.path.join(case_dir, "t_profile.dat")

    with open(filename, "w", encoding='utf-8') as file:
        for date in date_list:
            data_string_list = [f"{z}\t{t}" for z, t in zip(z_levels,t_profile)]
            data_string = "\n".join(data_string_list)
            file.write(f"{str(date)}\t{depth}\t{2}\n{data_string}\n")

    return filename


def edit_yaml(
        file: str,
        pattern: str,
        new_string: str
) -> None:
    
    with open(file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    line_to_edit = re.compile(pattern)

    for line in lines:
        match = line_to_edit.match(line)

        if match:
            indent = match.group('indent')
            new_line = (f"{indent}{new_string}\n")
            new_lines.append(new_line)
            continue
    
        new_lines.append(line)

    with open(file, 'w') as f:
        f.writelines(new_lines)
