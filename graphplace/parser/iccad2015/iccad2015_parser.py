from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..base_parser import BaseParser
from ..models import Design, Net, Node, Pin


@dataclass
class ICCAD2015FileSet:
    root: Path
    def_file: Optional[Path] = None
    lef_file: Optional[Path] = None
    verilog_file: Optional[Path] = None
    sdc_file: Optional[Path] = None
    meta_file: Optional[Path] = None


class ICCAD2015Parser(BaseParser):
    def collect_files(self) -> ICCAD2015FileSet:
        input_path = self.input_path
        root = input_path if input_path.is_dir() else input_path.parent

        files = ICCAD2015FileSet(root=root)
        if input_path.is_file():
            if input_path.suffix == ".def":
                files.def_file = input_path
            elif input_path.suffix == ".lef":
                files.lef_file = input_path
            elif input_path.suffix == ".v":
                files.verilog_file = input_path
            elif input_path.suffix == ".sdc":
                files.sdc_file = input_path
            elif input_path.suffix == ".iccad2015":
                files.meta_file = input_path

        if not files.def_file:
            def_candidates = list(root.glob("*.def"))
            if def_candidates:
                files.def_file = def_candidates[0]
        if not files.lef_file:
            lef_candidates = list(root.glob("*.lef"))
            if lef_candidates:
                files.lef_file = lef_candidates[0]
        if not files.verilog_file:
            v_candidates = list(root.glob("*.v"))
            if v_candidates:
                files.verilog_file = v_candidates[0]
        if not files.sdc_file:
            sdc_candidates = list(root.glob("*.sdc"))
            if sdc_candidates:
                files.sdc_file = sdc_candidates[0]
        if not files.meta_file:
            meta_candidates = list(root.glob("*.iccad2015"))
            if meta_candidates:
                files.meta_file = meta_candidates[0]

        if not files.def_file:
            raise FileNotFoundError("Missing .def file for ICCAD2015 dataset")

        return files

    def parse_design_name(self, files: ICCAD2015FileSet) -> str:
        if files.def_file:
            with files.def_file.open(mode="r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip().startswith("DESIGN"):
                        tokens = line.strip().replace(";", "").split()
                        if len(tokens) > 1:
                            return tokens[1]
        return files.root.name

    def parse_nodes(self, files: ICCAD2015FileSet, design: Design) -> None:
        if not files.def_file:
            return None
        self._parse_def_components(files.def_file, design)
        return None

    def parse_nets(self, files: ICCAD2015FileSet, design: Design) -> None:
        if not files.def_file:
            return None
        self._parse_def_nets(files.def_file, design)
        return None

    def parse_die(self, files: ICCAD2015FileSet, design: Design) -> None:
        if not files.def_file:
            return None
        self._parse_def_diearea(files.def_file, design)
        return None

    def _iter_def_lines(self, def_path: Path) -> Iterable[str]:
        with def_path.open(mode="r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                yield stripped

    def _parse_def_components(self, def_path: Path, design: Design) -> None:
        in_components = False
        buffer = ""

        for line in self._iter_def_lines(def_path):
            if line.startswith("COMPONENTS"):
                in_components = True
                continue
            if line.startswith("END COMPONENTS"):
                in_components = False
                buffer = ""
                continue
            if not in_components:
                continue

            buffer += " " + line
            if ";" not in line:
                continue
            entry = buffer.strip()
            buffer = ""

            match = re.search(r"-\s+(\S+)\s+(\S+)", entry)
            if not match:
                continue
            name = match.group(1)
            master = match.group(2)
            is_fixed = "FIXED" in entry
            coord_match = re.search(r"\(\s*(\d+)\s+(\d+)\s*\)", entry)
            x = float(coord_match.group(1)) if coord_match else 0.0
            y = float(coord_match.group(2)) if coord_match else 0.0

            node = Node(name=name, master=master, type="STD_CELL", x=x, y=y, is_fixed=is_fixed)
            design.add_node(node)

    def _parse_def_nets(self, def_path: Path, design: Design) -> None:
        in_nets = False
        buffer = ""

        for line in self._iter_def_lines(def_path):
            if line.startswith("NETS"):
                in_nets = True
                continue
            if line.startswith("END NETS"):
                in_nets = False
                buffer = ""
                continue
            if not in_nets:
                continue

            buffer += " " + line
            if ";" not in line:
                continue
            entry = buffer.strip()
            buffer = ""

            match = re.search(r"-\s+(\S+)", entry)
            if not match:
                continue
            net_name = match.group(1)
            net = Net(name=net_name)

            for node_name, pin_name in re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", entry):
                net.pins.append(Pin(node_name=node_name, name=pin_name))
                if node_name not in design.nodes and node_name.upper().startswith("PIN"):
                    design.add_node(Node(name=node_name, type="PORT", is_terminal=True))

            design.add_net(net)

    def _parse_def_diearea(self, def_path: Path, design: Design) -> None:
        buffer = ""
        diearea_re = re.compile(
            r"DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)"
        )

        with def_path.open(mode="r", encoding="utf-8") as handle:
            for line in handle:
                buffer += line.strip() + " "
                if ";" not in line:
                    continue
                match = diearea_re.search(buffer)
                if match:
                    design.die_lx = float(match.group(1))
                    design.die_ly = float(match.group(2))
                    design.die_hx = float(match.group(3))
                    design.die_hy = float(match.group(4))
                    return
                buffer = ""