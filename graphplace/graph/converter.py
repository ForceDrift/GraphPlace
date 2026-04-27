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
        # Add all nodes with their properties
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
            
        # Add Star Model edges (Net nodes)
        for net in design.nets:
            net_node_id = f"__net_{net.name}__"
            # Place the net node at the average position of its pins
            pin_nodes = [p.node_name for p in net.pins if p.node_name in design.nodes]
            if not pin_nodes:
                continue
                
            avg_x = sum(design.nodes[n].x for n in pin_nodes) / len(pin_nodes)
            avg_y = sum(design.nodes[n].y for n in pin_nodes) / len(pin_nodes)
            
            G.add_node(net_node_id, is_net=True, x=avg_x, y=avg_y)
            for node_name in pin_nodes:
                G.add_edge(node_name, net_node_id)
                
        return G
