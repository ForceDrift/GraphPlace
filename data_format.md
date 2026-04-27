#iccad 2015 format: 

Uses a set of files for incremental placement benchmarks:
- `.def`: Design Exchange Format (cell placement and connectivity)
- `.lef`: Library Exchange Format (cell physical dimensions)
- `.v`: Verilog netlist
- `.sdc`: Synopsys Design Constraints (timing constraints)
- `.lib`: Liberty timing library files
- `.iccad2015`: Metadata/configuration file for the benchmark

#ispd 2005 format: 

Standard Bookshelf format files:
- `.aux`: Auxiliary file listing all other benchmark files
- `.nodes`: Cell dimensions and terminal status
- `.nets`: Netlist connectivity (Pins and Nets)
- `.pl`: Placement coordinates for each node
- `.scl`: Row definitions for the placement area
- `.wts`: Weighted nets/nodes information

#ispd 2026 format: 

Uses a hybrid format of CSV and standard industry files:
- `node.csv`: Lower-left coordinates and master cell information for instances.
  - Header: `Name,Master,Type,llx,lly`
- `nets.csv`: Netlist connectivity in a custom CSV format.
  - Each line: `net_name,inst1 pin1,inst2 pin2, ...`
  - `_IO_` suffix identifies IO ports.
- `contest.def`: Standard DEF file for design structure and initial placement.
- `contest.sdc`: Timing constraints in Synopsys Design Constraints format.
- `contest.v`: Verilog netlist for connectivity.