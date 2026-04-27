from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from ..base_parser import BaseParser
from ..models import Design, Net, Node, Pin


@dataclass
class ISPDFileSet:
    kind: str
    root: Path
    aux: Optional[Path] = None
    nodes: Optional[Path] = None
    nets: Optional[Path] = None
    pl: Optional[Path] = None
    scl: Optional[Path] = None
    node_csv: Optional[Path] = None
    nets_csv: Optional[Path] = None
    def_file: Optional[Path] = None


class ISPDParser(BaseParser):
    def collect_files(self) -> ISPDFileSet:
        input_path = self.input_path
        root = input_path if input_path.is_dir() else input_path.parent

        if input_path.is_file() and input_path.suffix == ".aux":
            files = ISPDFileSet(kind="ispd2005", root=root, aux=input_path)
        elif (root / "node.csv").exists() or (input_path.name == "node.csv"):
            files = ISPDFileSet(kind="ispd2026", root=root)
        else:
            aux_files = list(root.glob("*.aux"))
            if aux_files:
                files = ISPDFileSet(kind="ispd2005", root=root, aux=aux_files[0])
            else:
                files = ISPDFileSet(kind="ispd2026", root=root)

        if files.kind == "ispd2005":
            if not files.aux:
                aux_candidates = list(files.root.glob("*.aux"))
                if aux_candidates:
                    files.aux = aux_candidates[0]
            if not files.aux:
                raise FileNotFoundError("Missing .aux file for ISPD2005 dataset")

            aux_map = self._parse_aux_file(files.aux)
            files.nodes = aux_map.get(".nodes")
            files.nets = aux_map.get(".nets")
            files.pl = aux_map.get(".pl")
            files.scl = aux_map.get(".scl")
        else:
            files.node_csv = files.root / "node.csv"
            files.nets_csv = files.root / "nets.csv"
            files.def_file = files.root / "contest.def"

        return files

    def parse_design_name(self, files: ISPDFileSet) -> str:
        if files.kind == "ispd2005":
            return files.aux.stem if files.aux else files.root.name
        if files.root.parent.name:
            return files.root.parent.name
        return files.root.name

    def parse_nodes(self, files: ISPDFileSet, design: Design) -> None:
        if files.kind == "ispd2005":
            if not files.nodes:
                raise FileNotFoundError("Missing .nodes file for ISPD2005 dataset")
            self._parse_bookshelf_nodes(files.nodes, design)
            if files.pl:
                self._parse_bookshelf_pl(files.pl, design)
            return None

        if not files.node_csv or not files.node_csv.exists():
            raise FileNotFoundError("Missing node.csv for ISPD2026 dataset")

        with files.node_csv.open(mode="r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = row.get("Name", "").strip()
                if not name:
                    continue
                node_type = row.get("Type", "").strip()
                is_terminal = node_type.upper() in {"PORT", "IO", "PIN"} or "_IO_" in name
                node = Node(
                    name=name,
                    master=row.get("Master", "").strip(),
                    type=node_type,
                    x=self._parse_float(row.get("llx"), 0.0),
                    y=self._parse_float(row.get("lly"), 0.0),
                    is_terminal=is_terminal,
                )
                design.add_node(node)

        return None

    def parse_nets(self, files: ISPDFileSet, design: Design) -> None:
        if files.kind == "ispd2005":
            if files.nets:
                self._parse_bookshelf_nets(files.nets, design)
            return None

        if not files.nets_csv or not files.nets_csv.exists():
            return None

        with files.nets_csv.open(mode="r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                net_name = row[0].strip()
                if not net_name or net_name.lower() in {"net", "net_name"}:
                    continue
                net = Net(name=net_name)
                for pin_entry in row[1:]:
                    pin_entry = pin_entry.strip()
                    if not pin_entry:
                        continue
                    tokens = pin_entry.split()
                    node_name = tokens[0]
                    pin_name = tokens[1] if len(tokens) > 1 else None
                    net.pins.append(Pin(node_name=node_name, name=pin_name))
                    if node_name not in design.nodes and "_IO_" in node_name:
                        design.add_node(Node(name=node_name, type="PORT", is_terminal=True))
                design.add_net(net)

        return None

    def parse_die(self, files: ISPDFileSet, design: Design) -> None:
        if files.kind == "ispd2005":
            if files.scl:
                self._parse_bookshelf_scl(files.scl, design)
            return None

        if files.def_file and files.def_file.exists():
            self._parse_def_diearea(files.def_file, design)
        return None

    def _parse_aux_file(self, aux_path: Path) -> Dict[str, Path]:
        with aux_path.open(mode="r", encoding="utf-8") as handle:
            content = handle.read().strip().split()

        file_map: Dict[str, Path] = {}
        for token in content:
            if token.endswith(":"):
                continue
            if token.startswith("RowBasedPlacement"):
                continue
            if token.startswith("#"):
                continue
            path = (aux_path.parent / token).resolve()
            file_map[path.suffix] = path
        return file_map

    def _parse_float(self, value: Optional[str], default: float) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _iter_bookshelf_lines(self, path: Path) -> Iterable[str]:
        with path.open(mode="r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("UCLA"):
                    continue
                yield stripped

    def _parse_bookshelf_nodes(self, nodes_path: Path, design: Design) -> None:
        for line in self._iter_bookshelf_lines(nodes_path):
            if line.startswith("NumNodes") or line.startswith("NumTerminals"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            name = tokens[0]
            width = float(tokens[1])
            height = float(tokens[2])
            is_terminal = len(tokens) > 3 and tokens[3].startswith("terminal")
            node_type = "PORT" if is_terminal else "STD_CELL"
            design.add_node(
                Node(
                    name=name,
                    width=width,
                    height=height,
                    type=node_type,
                    is_terminal=is_terminal,
                )
            )

    def _parse_bookshelf_pl(self, pl_path: Path, design: Design) -> None:
        for line in self._iter_bookshelf_lines(pl_path):
            if line.startswith("NumNodes"):
                continue
            cleaned = line.replace(":", " ")
            tokens = cleaned.split()
            if len(tokens) < 3:
                continue
            name = tokens[0]
            try:
                x = float(tokens[1])
                y = float(tokens[2])
            except ValueError:
                continue
            node = design.nodes.get(name, Node(name=name))
            node.x = x
            node.y = y
            if any(token.startswith("FIXED") for token in tokens[3:]):
                node.is_fixed = True
            design.add_node(node)

    def _parse_bookshelf_nets(self, nets_path: Path, design: Design) -> None:
        lines = list(self._iter_bookshelf_lines(nets_path))
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("NumNets") or line.startswith("NumPins"):
                idx += 1
                continue
            if line.startswith("NetDegree"):
                cleaned = line.replace(":", " ")
                tokens = cleaned.split()
                if len(tokens) < 3:
                    idx += 1
                    continue
                degree = int(tokens[1])
                net_name = tokens[2]
                net = Net(name=net_name)
                for offset in range(1, degree + 1):
                    if idx + offset >= len(lines):
                        break
                    pin_line = lines[idx + offset].replace(":", " ")
                    pin_tokens = pin_line.split()
                    if not pin_tokens:
                        continue
                    node_name = pin_tokens[0]
                    if len(pin_tokens) >= 4:
                        pin_name = pin_tokens[1]
                        offset_x = float(pin_tokens[-2])
                        offset_y = float(pin_tokens[-1])
                    elif len(pin_tokens) == 3:
                        pin_name = None
                        offset_x = float(pin_tokens[1])
                        offset_y = float(pin_tokens[2])
                    else:
                        pin_name = None
                        offset_x = 0.0
                        offset_y = 0.0
                    net.pins.append(Pin(node_name=node_name, name=pin_name, offset_x=offset_x, offset_y=offset_y))
                    if node_name not in design.nodes and "_IO_" in node_name:
                        design.add_node(Node(name=node_name, type="PORT", is_terminal=True))
                design.add_net(net)
                idx += degree + 1
                continue
            idx += 1

    def _parse_bookshelf_scl(self, scl_path: Path, design: Design) -> None:
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        current_y = None
        current_height = None
        site_width = None

        for line in self._iter_bookshelf_lines(scl_path):
            cleaned = line.replace(":", " ")
            tokens = cleaned.split()
            if not tokens:
                continue
            if tokens[0] == "Coordinate":
                current_y = float(tokens[1])
            elif tokens[0] == "Height":
                current_height = float(tokens[1])
            elif tokens[0] == "Sitewidth":
                site_width = float(tokens[1])
            elif tokens[0] == "SubrowOrigin":
                if current_y is None or current_height is None or site_width is None:
                    continue
                origin = float(tokens[1])
                num_sites = float(tokens[3])
                row_max_x = origin + num_sites * site_width
                min_x = min(min_x, origin)
                max_x = max(max_x, row_max_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y + current_height)

        if min_x != float("inf"):
            design.die_lx = min_x
            design.die_ly = min_y
            design.die_hx = max_x
            design.die_hy = max_y

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