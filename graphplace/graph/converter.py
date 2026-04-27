import networkx as nx
from ..parser.models import Design

class GraphConverter:
    """
    Converts a Design into a NetworkX graph representation.
    """
    
    @staticmethod
    def to_networkx(design: Design) -> nx.Graph:
        """
        Creates a graph and populates it with nodes from the design.
        """
        G = nx.Graph()
        
        # Add all nodes with their properties (x, y, width, height, etc.)
        for node_name, node in design.nodes.items():
            G.add_node(
                node_name, 
                x=node.x, 
                y=node.y, 
                width=node.width, 
                height=node.height,
                is_fixed=node.is_fixed,
                type=node.type
            )
            
        return G
