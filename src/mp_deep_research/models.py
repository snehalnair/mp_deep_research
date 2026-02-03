
"""Data models for materials research."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DataSource(Enum):
    """Track data provenance."""
    DFT_GGA = "DFT-GGA"
    DFT_GGA_U = "DFT-GGA+U"
    DFT_R2SCAN = "DFT-r2SCAN"
    EXPERIMENTAL = "Experimental"
    ICSD = "ICSD-sourced"
    PREDICTED = "ML-predicted"


@dataclass
class MaterialCandidate:
    """A material candidate with tracked provenance."""
    material_id: str
    formula: str

    # Thermodynamic
    formation_energy: Optional[float] = None
    energy_above_hull: Optional[float] = None
    is_stable: bool = False

    # Electronic
    band_gap: Optional[float] = None
    is_metal: bool = False
    is_direct_gap: Optional[bool] = None

    # Structural
    space_group: Optional[str] = None
    space_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    volume: Optional[float] = None
    density: Optional[float] = None
    nsites: Optional[int] = None

    # Provenance
    structure_source: str = "DFT-GGA"
    is_theoretical: bool = True

    # Screening
    rank_score: float = 0.0
    exclusion_reason: Optional[str] = None


@dataclass
class ScreeningResult:
    """Result from batch screening."""
    chemsys: str
    total_queried: int
    total_passed: int
    candidates: List[MaterialCandidate] = field(default_factory=list)
    exclusion_stats: Dict[str, int] = field(default_factory=dict)
    cache_path: Optional[str] = None
