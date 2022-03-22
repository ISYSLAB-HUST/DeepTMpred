#!/bin/sh
cd-hit -i $1 -o db_90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8  -d 0 -p 1 -T 16 -M 0 > db_90.log
cd-hit -i db_90 -o db_60 -c 0.6 -n 4 -g 1 -G 0 -aS 0.8  -d 0 -p 1 -T 16 -M 0 > db_60.log
psi-cd-hit.pl -i db_60 -o db_30 -c 0.3 > db_30.log
