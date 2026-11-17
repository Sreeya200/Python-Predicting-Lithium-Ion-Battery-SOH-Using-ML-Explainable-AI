import numpy as np
import pandas as pd
from scipy.io import loadmat
from src.config import BATTERY_FILES, DISCHARGE_CUTOFFS


# MATLAB helper functions
def get_field(obj, field):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    if hasattr(obj, field):
        return getattr(obj, field)
    return None


def numeric_vector(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(value).squeeze()
        arr = np.asarray(arr, dtype=float).flatten()
        return arr[np.isfinite(arr)]
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def get_cycle_type(cycle):
    value = get_field(cycle, "type")
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.squeeze()
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    return str(value).strip().lower()


def extract_cycle_data(cycle):
    data = get_field(cycle, "data")

    if data is None:
        empty = np.array([], dtype=float)
        return {
            "voltage": empty, "current": empty, "temperature": empty,
            "time": empty, "capacity": empty,
        }

    return {
        "voltage": numeric_vector(get_field(data, "Voltage_measured")),
        "current": numeric_vector(get_field(data, "Current_measured")),
        "temperature": numeric_vector(get_field(data, "Temperature_measured")),
        "time": numeric_vector(get_field(data, "Time")),
        "capacity": numeric_vector(get_field(data, "Capacity")),
    }


def get_cycles(battery):
    cycles = get_field(battery, "cycle")
    if cycles is None:
        raise ValueError("No 'cycle' field found in battery structure")
    return np.atleast_1d(cycles)


# Loading
def load_battery_objects(battery_files=None):
    battery_files = battery_files or BATTERY_FILES

    for battery_id, file_path in battery_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing NASA file: {file_path}")
    battery_objects = {}
    for battery_id, file_path in battery_files.items():
        mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)

        # NASA files normally use the battery ID as the top-level key
        battery = mat_data.get(battery_id)

        # Fall back to the only available non-system key
        if battery is None:
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if len(keys) == 1:
                battery = mat_data[keys[0]]

        if battery is None:
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            raise ValueError(
                f"Battery structure not found in {file_path}.\n"
                f"Available keys: {keys}"
            )

        battery_objects[battery_id] = battery

    return battery_objects


# Stage 1: raw inspection
def inspect_batteries(battery_objects):
    battery_summary = []
    discharge_inventory = []
    matlab_structure = {}
    for battery_id, battery in battery_objects.items():
        cycles = get_cycles(battery)
        charge_count = discharge_count = impedance_count = unknown_count = 0
        discharge_records = []
        for cycle_number, cycle in enumerate(cycles, start=1):
            cycle_type = get_cycle_type(cycle)
            if cycle_type == "charge":
                charge_count += 1
            elif cycle_type == "discharge":
                discharge_count += 1
                signals = extract_cycle_data(cycle)
                voltage = signals["voltage"]
                temperature = signals["temperature"]
                time = signals["time"]
                capacity = signals["capacity"]
                valid_capacity = capacity[np.isfinite(capacity)]
                final_capacity = (
                    valid_capacity[-1] if len(valid_capacity) > 0 else np.nan
                )
                duration = time[-1] - time[0] if len(time) >= 2 else np.nan
                discharge_records.append({
                    "battery_id": battery_id,
                    "cycle_number": cycle_number,
                    "capacity_ah": final_capacity,
                    "duration_seconds": duration,
                    "voltage_min": np.min(voltage) if len(voltage) else np.nan,
                    "voltage_max": np.max(voltage) if len(voltage) else np.nan,
                    "temperature_min":
                        np.min(temperature) if len(temperature) else np.nan,
                    "temperature_max":
                        np.max(temperature) if len(temperature) else np.nan,
                    "measurement_count": len(voltage),
                })

            elif cycle_type == "impedance":
                impedance_count += 1
            else:
                unknown_count += 1

        # Capacity fade from the first and last valid discharge
        valid_capacities = [
            r["capacity_ah"] for r in discharge_records
            if pd.notna(r["capacity_ah"])
        ]
        first_capacity = valid_capacities[0] if valid_capacities else np.nan
        last_capacity = valid_capacities[-1] if valid_capacities else np.nan

        if (pd.notna(first_capacity) and first_capacity != 0
                and pd.notna(last_capacity)):
            capacity_fade = (first_capacity - last_capacity) / first_capacity * 100
        else:
            capacity_fade = np.nan

        battery_summary.append({
            "battery_id": battery_id,
            "total_cycles": len(cycles),
            "charge_cycles": charge_count,
            "discharge_cycles": discharge_count,
            "impedance_cycles": impedance_count,
            "unknown_cycles": unknown_count,
            "first_capacity_ah": first_capacity,
            "last_capacity_ah": last_capacity,
            "capacity_fade_percent": capacity_fade,
            "documented_cutoff_voltage": DISCHARGE_CUTOFFS[battery_id],
        })

        discharge_inventory.extend(discharge_records)

        matlab_structure[battery_id] = {
            "number_of_cycles": len(cycles),
            "cycle_types": {
                "charge": charge_count,
                "discharge": discharge_count,
                "impedance": impedance_count,
                "unknown": unknown_count,
            },
        }

    return (
        pd.DataFrame(battery_summary),
        pd.DataFrame(discharge_inventory),
        matlab_structure,
    )


# Stage 2: cycle-level SoH dataset
def build_cycle_dataset(battery_objects):
    cycle_records = []
    for battery_id, battery in battery_objects.items():
        cycles = get_cycles(battery)
        latest_charge = None
        latest_charge_number = None
        for cycle_number, cycle in enumerate(cycles, start=1):
            cycle_type = get_cycle_type(cycle)
            if cycle_type == "charge":
                latest_charge = extract_cycle_data(cycle)
                latest_charge_number = cycle_number
            elif cycle_type == "discharge":
                discharge = extract_cycle_data(cycle)
                valid_capacity = discharge["capacity"][
                    np.isfinite(discharge["capacity"])
                ]
                if len(valid_capacity) == 0:
                    continue
                final_capacity = valid_capacity[-1]

                dis_time = discharge["time"]
                discharge_duration = (
                    dis_time[-1] - dis_time[0] if len(dis_time) >= 2 else np.nan
                )
                if latest_charge is not None:
                    chg_time = latest_charge["time"]
                    charge_duration = (
                        chg_time[-1] - chg_time[0]
                        if len(chg_time) >= 2 else np.nan
                    )
                    charge_voltage_count = len(latest_charge["voltage"])
                    charge_current_count = len(latest_charge["current"])
                    charge_temperature_count = len(latest_charge["temperature"])
                else:
                    charge_duration = np.nan
                    charge_voltage_count = 0
                    charge_current_count = 0
                    charge_temperature_count = 0

                cycle_records.append({
                    "battery_id": battery_id,
                    "cycle_number": cycle_number,
                    "charge_cycle_number": latest_charge_number,
                    "capacity_ah": final_capacity,
                    "charge_duration": charge_duration,
                    "discharge_duration": discharge_duration,
                    "charge_voltage_count": charge_voltage_count,
                    "charge_current_count": charge_current_count,
                    "charge_temperature_count": charge_temperature_count,
                    "discharge_voltage_count": len(discharge["voltage"]),
                    "discharge_current_count": len(discharge["current"]),
                    "discharge_temperature_count": len(discharge["temperature"]),
                })

    cycle_data = pd.DataFrame(cycle_records)

    cycle_data = cycle_data.sort_values(
        ["battery_id", "cycle_number"]
    ).reset_index(drop=True)

    cycle_data["initial_capacity_ah"] = (
        cycle_data.groupby("battery_id")["capacity_ah"].transform("first")
    )
    cycle_data["soh_relative"] = (
        cycle_data["capacity_ah"] / cycle_data["initial_capacity_ah"]
    )
    cycle_data["discharge_sequence"] = (
        cycle_data.groupby("battery_id").cumcount() + 1
    )

    return cycle_data
