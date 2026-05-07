replace_external rep
rep import_lef /data/externals/temp/replace_ibm01/ibm01.lef
rep import_def /data/externals/temp/replace_ibm01/ibm01.def
rep set_output /data/externals/temp/replace_ibm01/output/
rep set_density 1.0
rep init_replace
rep place_cell_nesterov_place
rep export_def /data/externals/temp/replace_ibm01/ibm01.out.def
exit
