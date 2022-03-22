#!/bin/bash
DB="UniRef30_2020_03"
query="input squence"
outfile="HMM proflie ouput"
hhblits -i $query -d $DB -n 3 -e 0.001 -cpu 14 -ohhm $outfile -v 0 -Z 0