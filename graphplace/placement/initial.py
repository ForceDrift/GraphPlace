import random
from ..parser.models import Design

class InitialPlacement:
    """
    Utilities for initial placement of nodes.
    Useful for starting points in RL or analytical placement.
    """
    
    @staticmethod
    def random_placement(design: Design):
        """
        Places all non-fixed nodes at random locations within the die area.
        """
        width_range = design.die_hx - design.die_lx
        height_range = design.die_hy - design.die_ly
        
        for node in design.nodes.values():
            if not node.is_fixed:
                # Randomize lower-left corner
                # We subtract width/height to keep the node inside the boundary
                max_x = max(design.die_lx, design.die_hx - node.width)
                max_y = max(design.die_ly, design.die_hy - node.height)
                
                node.x = random.uniform(design.die_lx, max_x)
                node.y = random.uniform(design.die_ly, max_y)

    @staticmethod
    def center_placement(design: Design):
        """
        Places all non-fixed nodes at the center of the die.
        Common starting point for analytical force-directed placement.
        """
        center_x = (design.die_lx + design.die_hx) / 2
        center_y = (design.die_ly + design.die_hy) / 2
        
        for node in design.nodes.values():
            if not node.is_fixed:
                node.x = center_x - (node.width / 2)
                node.y = center_y - (node.height / 2)
