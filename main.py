from graphplace.parser.ispd.ispd_parser import ISPDParser
from graphplace.placement.initial import InitialPlacement
from graphplace.graph.converter import GraphConverter
from graphplace.visualization.plotter import DesignPlotter
from pathlib import Path

def main():
    # Path to the ISPD 2005 sample .aux file
    input_path = Path("data/ispd2005_sample/adaptec1.inf.aux")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Parsing {input_path}...")
    
    # Initialize the parser
    parser = ISPDParser(str(input_path))
    
    # Parse the design
    design = parser.parse()
    
    # 1. Initialize Randomly
    print("Randomizing node positions...")
    InitialPlacement.random_placement(design)
    
    # 2. Convert to NetworkX Graph
    print("Initializing graph with nodes...")
    G = GraphConverter.to_networkx(design)
    
    print("\n--- Graph Initialized ---")
    print(f"Number of nodes in graph: {G.number_of_nodes()}")
    
    # 3. Visualize Graph
    print("\nGenerating graph visualization...")
    DesignPlotter.plot_graph(G, "graph_plot.png")
    
    # 4. Visualize Placement
    print("\nGenerating visualization...")
    DesignPlotter.plot_design(design, "placement.png")

if __name__ == "__main__":
    main()