# RePlAce Configuration for ibm01
set_output "/Users/thanosanp/Documents/GraphPlace/data/ibm01_output"
set_density 0.7
# Bookshelf files are in /data
import_lef "/data/ibm01.nodes"
# Actually, RePlAce standalone reading Bookshelf uses a different flow
# or it can take the .aux file directly.
