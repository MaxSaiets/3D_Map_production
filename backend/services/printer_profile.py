from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class PrinterProfile:
    printer_model: str
    nozzle_diameter_mm: float
    nominal_line_width_mm: float
    external_line_width_mm: float
    first_layer_line_width_mm: float
    groove_side_clearance_mm: float
    min_printable_feature_mm: float
    tiny_cleanup_mm: float
    road_gap_fill_mm: float
    acute_corner_collapse_mm: float
    terrain_boolean_overcut_mm: float
    elephant_foot_compensation_mm: float

    def with_overrides(
        self,
        *,
        nozzle_diameter_mm: float | None = None,
        groove_side_clearance_mm: float | None = None,
        min_printable_feature_mm: float | None = None,
        road_gap_fill_mm: float | None = None,
        elephant_foot_compensation_mm: float | None = None,
    ) -> "PrinterProfile":
        updated = self
        if nozzle_diameter_mm is not None and nozzle_diameter_mm > 0:
            updated = replace(updated, nozzle_diameter_mm=float(nozzle_diameter_mm))
        if groove_side_clearance_mm is not None and groove_side_clearance_mm > 0:
            updated = replace(updated, groove_side_clearance_mm=float(groove_side_clearance_mm))
        if min_printable_feature_mm is not None and min_printable_feature_mm > 0:
            updated = replace(updated, min_printable_feature_mm=float(min_printable_feature_mm))
        if road_gap_fill_mm is not None and road_gap_fill_mm > 0:
            updated = replace(updated, road_gap_fill_mm=float(road_gap_fill_mm))
        if elephant_foot_compensation_mm is not None and elephant_foot_compensation_mm >= 0:
            updated = replace(updated, elephant_foot_compensation_mm=float(elephant_foot_compensation_mm))
        return updated


DEFAULT_BAMBU_P1S_04 = PrinterProfile(
    printer_model="bambu_lab_p1s",
    nozzle_diameter_mm=0.4,
    nominal_line_width_mm=0.45,
    external_line_width_mm=0.42,
    first_layer_line_width_mm=0.45,
    groove_side_clearance_mm=0.15,
    # Canonical mask QA is intentionally stricter than the single-line nozzle
    # minimum: sub-0.6mm islands, bays, and holes print unreliably and look
    # ragged on park/water/terrain boundaries. Road topology remains protected
    # by faithful road handling in runtime_canonical_masks.
    min_printable_feature_mm=0.6,
    tiny_cleanup_mm=0.2,
    road_gap_fill_mm=0.3,
    acute_corner_collapse_mm=0.6,
    terrain_boolean_overcut_mm=0.03,
    elephant_foot_compensation_mm=0.2,
)


def get_default_printer_profile() -> PrinterProfile:
    return DEFAULT_BAMBU_P1S_04


def get_printer_profile_for_request(request: Any | None) -> PrinterProfile:
    profile = get_default_printer_profile()
    if request is None:
        return profile

    nozzle = getattr(request, "nozzle_diameter_mm", None)
    groove = getattr(request, "groove_side_clearance_mm", None)
    min_feature = getattr(request, "min_printable_feature_mm", None)
    road_gap = getattr(request, "road_gap_fill_threshold_mm", None)
    elephant = getattr(request, "elephant_foot_compensation_mm", None)
    return profile.with_overrides(
        nozzle_diameter_mm=float(nozzle) if nozzle not in (None, "") else None,
        groove_side_clearance_mm=float(groove) if groove not in (None, "") else None,
        min_printable_feature_mm=float(min_feature) if min_feature not in (None, "") else None,
        road_gap_fill_mm=float(road_gap) if road_gap not in (None, "") else None,
        elephant_foot_compensation_mm=float(elephant) if elephant not in (None, "") else None,
    )
