from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Pin:
    node_name: str
    offset_x: float = 0.0
    offset_y: float = 0.0
    name: Optional[str] = None

@dataclass
class Node:
    name: str
    master: str = ""        # Master cell type (e.g. 'NAND2_X1')
    type: str = ""          # Category (e.g. 'STD_CELL', 'MACRO', 'PORT')
    width: float = 0.0
    height: float = 0.0
    x: float = 0.0          # Lower-left X coordinate
    y: float = 0.0          # Lower-left Y coordinate
    is_fixed: bool = False  # Fixed nodes cannot be moved during placement
    is_terminal: bool = False # For Bookshelf format support


@dataclass
class Net:
    name: str
    pins: List[Pin] = field(default_factory=list)
    weight: float = 1.0

@dataclass
class Design:
    name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    nets: List[Net] = field(default_factory=list)
    
    die_lx: float = 0.0
    die_ly: float = 0.0
    die_hx: float = 0.0
    die_hy: float = 0.0

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def add_net(self, net: Net):
        self.nets.append(net)
