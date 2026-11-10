from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


MODES = {"normalized_proxy", "physical_6s"}
COEFFICIENT_FIELDS = (
    "tau",
    "E_sun",
    "E_sky",
    "L_path",
    "response_weight",
    "delta_lambda",
)
NORMALIZED_SEMANTICS = "dimensionless_fixed_auxiliary_model_coefficients"
PHYSICAL_SEMANTICS = "physical_radiative_transfer_coefficients"
NORMALIZED_UNITS = {
    "tau": "dimensionless",
    "E_sun": "dimensionless",
    "E_sky": "dimensionless",
    "L_path": "dimensionless",
    "response_weight": "dimensionless",
    "delta_lambda": "dimensionless",
    "output": "dimensionless_radiance_related_score",
}
PHYSICAL_UNITS = {
    "tau": "dimensionless",
    "E_sun": "W m^-2 um^-1",
    "E_sky": "W m^-2 um^-1",
    "L_path": "W m^-2 sr^-1 um^-1",
    "response_weight": "dimensionless",
    "delta_lambda": "um",
    "output": "W m^-2 sr^-1",
}
PROVENANCE_FIELDS = (
    "sensor_id",
    "band_id",
    "wavelength_definition",
    "spectral_response_definition",
    "acquisition_date",
    "acquisition_time",
    "latitude_deg",
    "longitude_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "view_zenith_deg",
    "view_azimuth_deg",
    "target_altitude_km",
    "sensor_altitude_km",
    "atmospheric_profile",
    "aerosol_profile",
    "aot550_or_visibility",
    "water_vapor_g_cm2",
    "ozone_cm_atm",
    "additional_6s_inputs",
)


@dataclass(frozen=True)
class RTDContext:
    """Validated future-loader payload; it is not connected to training."""

    mode: str
    context_id: str
    sensor_id: str
    target_band: str
    wavelength_um: tuple[float, ...] | None
    wavelength_range_um: tuple[float, float] | None
    tau: tuple[float, ...]
    E_sun: tuple[float, ...]
    E_sky: tuple[float, ...]
    L_path: tuple[float, ...]
    response_weight: tuple[float, ...]
    delta_lambda: tuple[float, ...]
    eps: float
    provenance: Mapping[str, Any] | None
    raw: Mapping[str, Any]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_sequence(value: object, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a non-empty list of finite numbers")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"provenance_file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, "physical provenance")


def _provenance_leaf(provenance: Mapping[str, Any], field: str) -> dict[str, Any]:
    leaf = _mapping(provenance.get(field), f"provenance.{field}")
    if "value" not in leaf or leaf["value"] is None or leaf["value"] == "":
        raise ValueError(f"provenance.{field}.value must be explicit")
    _text(leaf.get("source"), f"provenance.{field}.source")
    return leaf


def _validate_physical_provenance(
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    leaves = {field: _provenance_leaf(provenance, field) for field in PROVENANCE_FIELDS}
    if str(leaves["sensor_id"]["value"]) != config["sensor_id"]:
        raise ValueError("sensor/config mismatch between physical provenance and context")
    if str(leaves["band_id"]["value"]) != config["target_band"]:
        raise ValueError("band/config mismatch between physical provenance and context")

    wavelength_definition = _mapping(
        leaves["wavelength_definition"]["value"],
        "provenance.wavelength_definition.value",
    )
    if config.get("wavelength_um") is not None:
        recorded = wavelength_definition.get("wavelength_um")
        recorded_values = _finite_sequence(recorded, "provenance wavelength_um")
        expected = tuple(float(item) for item in config["wavelength_um"])
        if recorded_values != expected:
            raise ValueError("wavelength/config mismatch between physical provenance and context")
    else:
        recorded = wavelength_definition.get("wavelength_range_um")
        recorded_values = _finite_sequence(recorded, "provenance wavelength_range_um")
        expected = tuple(float(item) for item in config["wavelength_range_um"])
        if recorded_values != expected:
            raise ValueError("wavelength/config mismatch between physical provenance and context")

    latitude = _finite(leaves["latitude_deg"]["value"], "provenance latitude_deg")
    longitude = _finite(leaves["longitude_deg"]["value"], "provenance longitude_deg")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("provenance latitude_deg must be within [-90, 90]")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("provenance longitude_deg must be within [-180, 180]")
    for field in ("solar_zenith_deg", "view_zenith_deg"):
        value = _finite(leaves[field]["value"], f"provenance {field}")
        if not 0.0 <= value <= 180.0:
            raise ValueError(f"provenance {field} must be within [0, 180]")
    for field in ("solar_azimuth_deg", "view_azimuth_deg"):
        value = _finite(leaves[field]["value"], f"provenance {field}")
        if not 0.0 <= value < 360.0:
            raise ValueError(f"provenance {field} must be within [0, 360)")


def _validate_wavelength(config: dict[str, Any]) -> int:
    has_point = config.get("wavelength_um") is not None
    has_range = config.get("wavelength_range_um") is not None
    if has_point == has_range:
        raise ValueError("exactly one of wavelength_um or wavelength_range_um is required")
    if has_point:
        values = _finite_sequence(config["wavelength_um"], "wavelength_um")
        if any(value <= 0.0 for value in values):
            raise ValueError("wavelength_um values must be positive")
        config["wavelength_um"] = list(values)
        return len(values)
    values = _finite_sequence(config["wavelength_range_um"], "wavelength_range_um")
    if len(values) != 2 or values[0] <= 0.0 or values[0] >= values[1]:
        raise ValueError("wavelength_range_um must be [positive_min, larger_max]")
    config["wavelength_range_um"] = list(values)
    return 1


def validate_rtd_context(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    expected_sensor_id: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize an RTD context without connecting it to training."""

    config = copy.deepcopy(_mapping(payload, "RTD context"))
    mode = _text(config.get("mode"), "mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if config.get("template") is True:
        raise ValueError("template RTD contexts are intentionally not loadable as final contexts")

    for field in (
        "context_id",
        "sensor_id",
        "target_band",
        "spectral_response_source",
        "parameter_source",
        "generation_method",
        "parameter_semantics",
    ):
        config[field] = _text(config.get(field), field)
    if expected_sensor_id is not None and config["sensor_id"] != expected_sensor_id:
        raise ValueError("sensor/config mismatch for requested RTD context")

    band_count = _validate_wavelength(config)
    coefficients: dict[str, tuple[float, ...]] = {}
    for field in COEFFICIENT_FIELDS:
        coefficients[field] = _finite_sequence(config.get(field), field)
        if len(coefficients[field]) != band_count:
            raise ValueError(f"{field} length must match the wavelength/band definition")
        config[field] = list(coefficients[field])

    if any(value < 0.0 or value > 1.0 for value in coefficients["tau"]):
        raise ValueError("tau values must be within [0, 1]")
    for field in ("E_sun", "E_sky", "L_path"):
        if any(value < 0.0 for value in coefficients[field]):
            raise ValueError(f"{field} values must be non-negative")
    if any(value < 0.0 for value in coefficients["response_weight"]):
        raise ValueError("response weights must be non-negative")
    if not math.isclose(sum(coefficients["response_weight"]), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("response weights must sum to 1")
    if any(value <= 0.0 for value in coefficients["delta_lambda"]):
        raise ValueError("delta_lambda values must be positive")
    config["eps"] = _finite(config.get("eps"), "eps")
    if config["eps"] <= 0.0:
        raise ValueError("eps must be positive")

    units = _mapping(config.get("units"), "units")
    expected_units = PHYSICAL_UNITS if mode == "physical_6s" else NORMALIZED_UNITS
    if units != expected_units:
        raise ValueError(f"{mode} units must exactly match the declared mode contract")

    provenance: dict[str, Any] | None = None
    if mode == "normalized_proxy":
        if config["parameter_semantics"] != NORMALIZED_SEMANTICS:
            raise ValueError("normalized_proxy cannot masquerade as physical_6s")
        if config.get("provenance_file") not in (None, ""):
            raise ValueError("normalized_proxy must not attach physical_6s provenance")
        if config.get("physical_gate") not in (None, {}):
            raise ValueError("normalized_proxy must not claim a physical_6s gate")
    else:
        if config["parameter_semantics"] != PHYSICAL_SEMANTICS:
            raise ValueError("physical_6s must declare physical coefficient semantics")
        provenance_file = _text(config.get("provenance_file"), "provenance_file")
        provenance_path = Path(provenance_file)
        if not provenance_path.is_absolute():
            provenance_path = Path(base_dir) / provenance_path
        provenance = _load_mapping(provenance_path.resolve())
        _validate_physical_provenance(provenance, config)

        gate = _mapping(config.get("physical_gate"), "physical_gate")
        required_gate = {
            "tooling": "PASS",
            "dataset_metadata": "VERIFIED",
            "parameter_mapping": "VERIFIED",
            "unit_consistency": "PASS",
            "double_counting": "PASS",
            "forward_consistency": "PASS",
        }
        if gate != required_gate:
            raise ValueError("physical_6s requires every provenance and scientific gate to pass")
        config["physical_provenance"] = provenance
    return config


def deterministic_context_json(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    expected_sensor_id: str | None = None,
) -> str:
    validated = validate_rtd_context(
        payload,
        base_dir=base_dir,
        expected_sensor_id=expected_sensor_id,
    )
    return json.dumps(
        validated,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def load_rtd_context(
    path: str | Path,
    *,
    expected_sensor_id: str | None = None,
) -> RTDContext:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return rtd_context_from_mapping(
        payload,
        base_dir=path.parent,
        expected_sensor_id=expected_sensor_id,
    )


def rtd_context_from_mapping(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    expected_sensor_id: str | None = None,
) -> RTDContext:
    """Build the future-loader payload from an already in-memory context mapping."""

    config = validate_rtd_context(
        payload,
        base_dir=base_dir,
        expected_sensor_id=expected_sensor_id,
    )
    wavelength_um = config.get("wavelength_um")
    wavelength_range_um = config.get("wavelength_range_um")
    return RTDContext(
        mode=config["mode"],
        context_id=config["context_id"],
        sensor_id=config["sensor_id"],
        target_band=config["target_band"],
        wavelength_um=tuple(wavelength_um) if wavelength_um is not None else None,
        wavelength_range_um=(
            tuple(wavelength_range_um) if wavelength_range_um is not None else None
        ),
        tau=tuple(config["tau"]),
        E_sun=tuple(config["E_sun"]),
        E_sky=tuple(config["E_sky"]),
        L_path=tuple(config["L_path"]),
        response_weight=tuple(config["response_weight"]),
        delta_lambda=tuple(config["delta_lambda"]),
        eps=config["eps"],
        provenance=config.get("physical_provenance"),
        raw=config,
    )
