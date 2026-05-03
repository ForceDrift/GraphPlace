import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Benchmark:
    """
    Placement benchmark in pure PyTorch tensors.

    All coordinates are in microns.
    All indices are 0-based.

    Tensors contain both hard macros (indices [0, num_hard_macros)) and
    soft macros (indices [num_hard_macros, num_macros)). Hard macros are
    the primary optimization targets; soft macros are standard cell clusters
    that should be co-optimized for best results.
    """

    name: str
    canvas_width: float
    canvas_height: float
    num_macros: int
    macro_positions: torch.Tensor
    macro_sizes: torch.Tensor
    macro_fixed: torch.Tensor
    macro_names: List[str]
    num_nets: int
    net_nodes: List[torch.Tensor]
    net_weights: torch.Tensor
    grid_rows: int
    grid_cols: int
    port_positions: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))
    macro_pin_offsets: List[torch.Tensor] = field(default_factory=list)
    net_pin_nodes: List[torch.Tensor] = field(default_factory=list)
    hroutes_per_micron: float = 11.285
    vroutes_per_micron: float = 12.605
    hard_macro_indices: List[int] = field(default_factory=list)
    soft_macro_indices: List[int] = field(default_factory=list)
    num_hard_macros: int = 0
    num_soft_macros: int = 0

    def __post_init__(self):
        if self.num_hard_macros == 0 and self.num_soft_macros == 0:
            self.num_hard_macros = self.num_macros
            self.num_soft_macros = 0

        assert self.num_macros == self.num_hard_macros + self.num_soft_macros, (
            f"num_macros {self.num_macros} != "
            f"num_hard {self.num_hard_macros} + num_soft {self.num_soft_macros}"
        )
        assert self.macro_positions.shape == (self.num_macros, 2), (
            f"macro_positions shape {self.macro_positions.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_sizes.shape == (self.num_macros, 2), (
            f"macro_sizes shape {self.macro_sizes.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_fixed.shape == (self.num_macros,), (
            f"macro_fixed shape {self.macro_fixed.shape} != ({self.num_macros},)"
        )

        if len(self.net_nodes) > 0:
            assert len(self.net_nodes) == self.num_nets, (
                f"len(net_nodes) {len(self.net_nodes)} != num_nets {self.num_nets}"
            )

        if len(self.net_pin_nodes) > 0:
            assert len(self.net_pin_nodes) == self.num_nets, (
                f"len(net_pin_nodes) {len(self.net_pin_nodes)} != num_nets {self.num_nets}"
            )

        assert self.net_weights.shape == (self.num_nets,), (
            f"net_weights shape {self.net_weights.shape} != ({self.num_nets},)"
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "name": self.name,
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "num_macros": self.num_macros,
                "num_hard_macros": self.num_hard_macros,
                "num_soft_macros": self.num_soft_macros,
                "macro_positions": self.macro_positions,
                "macro_sizes": self.macro_sizes,
                "macro_fixed": self.macro_fixed,
                "macro_names": self.macro_names,
                "num_nets": self.num_nets,
                "net_nodes": self.net_nodes,
                "net_weights": self.net_weights,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "hroutes_per_micron": self.hroutes_per_micron,
                "vroutes_per_micron": self.vroutes_per_micron,
                "port_positions": self.port_positions,
                "macro_pin_offsets": self.macro_pin_offsets,
                "net_pin_nodes": self.net_pin_nodes,
                "hard_macro_indices": self.hard_macro_indices,
                "soft_macro_indices": self.soft_macro_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        data = torch.load(path, weights_only=False)
        if "num_hard_macros" not in data:
            data["num_hard_macros"] = data["num_macros"]
            data["num_soft_macros"] = 0
        if "soft_macro_indices" not in data:
            data["soft_macro_indices"] = []
        if "port_positions" not in data:
            data["port_positions"] = torch.zeros(0, 2)
        if "macro_pin_offsets" not in data:
            data["macro_pin_offsets"] = []
        if "net_pin_nodes" not in data:
            data["net_pin_nodes"] = []
        return cls(**data)

    def get_movable_mask(self) -> torch.Tensor:
        return ~self.macro_fixed

    def get_hard_macro_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[: self.num_hard_macros] = True
        return mask

    def get_soft_macro_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[self.num_hard_macros :] = True
        return mask

    @property
    def x(self) -> torch.Tensor:
        return self.macro_positions[:, 0]

    @property
    def y(self) -> torch.Tensor:
        return self.macro_positions[:, 1]

    def __repr__(self) -> str:
        num_ports = self.port_positions.shape[0]
        return (
            "Benchmark(\n"
            f"  name='{self.name}',\n"
            f"  macros={{hard: {self.num_hard_macros}, soft: {self.num_soft_macros}}},\n"
            f"  nets={self.num_nets},\n"
            f"  ports={num_ports},\n"
            f"  canvas={self.canvas_width:.1f}x{self.canvas_height:.1f}um,\n"
            f"  grid={self.grid_rows}x{self.grid_cols},\n"
            f"  routing={{H: {self.hroutes_per_micron}, V: {self.vroutes_per_micron}}}\n"
            ")"
        )
