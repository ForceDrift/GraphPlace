from .models import Node
import csv
from typing import List

#todo get input path as foler or something then go from there

class ISPDParser:
    def __init__(self, input_path: str):
        self.nodes = self.parse_nodes(input_path)
        #ie --> aes_cypher, ariene , etc 
        self.design_type = self.get_design_name(input_path)
   
    def get_design_name(self, input_path: str) -> str: 
        with open(input_path, mode="r") as f:
            for line in f:
                if "DESIGN" in line:
                    return line.replace("DESIGN", "").replace(";", "").strip()

    #get nodes from benchmark 
    def parse_nodes(self, input_path: str) -> List[Node]:
        nodes = []
        with open(input_path, mode = "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if (self.design_type.startswith("ariene")):
                for row in reader:
                    node = Node(
                        name=row['Name'],
                        master=row['Master'],
                        type=row['Type'],
                        llx=int(row['llx']),
                        lly=int(row['lly'])
                    )
                nodes.append(node)
        return nodes
                    
            
                    
        

    

    
        


            
        
        
