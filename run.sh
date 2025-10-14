#!/bin/bash
# Activate venv (mac/linux). Edit for Windows if needed.
source .venv/bin/activate
python src/qft_encrypt.py --input data/sample_images/original.png --size 128 --mode all