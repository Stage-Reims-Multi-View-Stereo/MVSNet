#!/bin/bash

echo "***** LOADING MODULES"

# Singularity
module load singularity/3.7.1

# Pouvoir utiliser le GPU (pas forcément nécessaire)
module load pgi/20.11

# Tensorboard / Jupyter (ne fonctionne pas sans...)
module load openssl
module load libffi

echo "***** RUNNING COMMAND:"
echo "$@"

echo "***** COMMAND OUTPUT"
"$@"