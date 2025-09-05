#!/bin/bash

# Force bitsandbytes to load CUDA 12.8 shared library
export BITSANDBYTES_FORCE_CUDA_VERSION=128

# Start the Gunicorn server
exec gunicorn -w 1 --timeout 720 -b 0.0.0.0:6969 api.app:app
