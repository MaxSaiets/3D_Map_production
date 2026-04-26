from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InlayFitConfig:
    # Insert meshes stay canonical by default. We only widen grooves.
    insert_side_clearance_mm: float = 0.0
    groove_side_clearance_mm: float = 0.15

    def insert_clearance_m(self, scale_factor: float | None) -> float:
        if not scale_factor or scale_factor <= 0:
            return 0.0
        return (self.insert_side_clearance_mm / 1000.0) / float(scale_factor)

    def groove_clearance_m(self, scale_factor: float | None) -> float:
        if not scale_factor or scale_factor <= 0:
            return 0.0
        return (self.groove_side_clearance_mm / 1000.0) / float(scale_factor)
