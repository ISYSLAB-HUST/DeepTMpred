#!/bin/bash
DB="NR database"
query="input squence"
output="ouput"
outfile="PSSM ouput"
psiblast -query $query -db $DB -num_iterations 3 -evalue 0.001 -out $output -out_ascii_pssm $outfile
