#!/bin/bash

find ./ -type d -name "*_pycache_* " | xargs -I {} rm -rf {} \;

rm -rf /home/dw3zn/Desktop/Repos/hvae-oodd/notebooks/.ipynb_checkpoints

rm -rf *.out

