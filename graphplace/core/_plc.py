import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLC_CLIENT_CANDIDATES = [
    _PROJECT_ROOT / "externals" / "MacroPlacement" / "CodeElements" / "Plc_client",
    _PROJECT_ROOT / "external" / "MacroPlacement" / "CodeElements" / "Plc_client",
]

_PLC_CLIENT_DIR = None
for candidate in _PLC_CLIENT_CANDIDATES:
    if candidate.exists():
        _PLC_CLIENT_DIR = str(candidate)
        break

if _PLC_CLIENT_DIR is None:
    raise FileNotFoundError(
        "Could not find Plc_client directory under externals/ or external/."
    )

if _PLC_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _PLC_CLIENT_DIR)

from plc_client_os import PlacementCost  # noqa: E402  # type: ignore

__all__ = ["PlacementCost"]
