import sys
from pathlib import Path

_PLC_CLIENT_DIR = str(
    Path(__file__).resolve().parent.parent
    / "external"
    / "MacroPlacement"
    / "CodeElements"
    / "Plc_client"
)

if _PLC_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _PLC_CLIENT_DIR)

from plc_client_os import PlacementCost # noqa  # type: ignore
__all__ = ["PlacementCost"]