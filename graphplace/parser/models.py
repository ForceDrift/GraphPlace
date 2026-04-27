from dataclasses import dataclass
from typing import List

@dataclass

#nodes for ispd are like this --> change to a general definition for all .node files 
class Node:
    name: str 
    master: str 
    type: str
    llx: int 
    lly: int


