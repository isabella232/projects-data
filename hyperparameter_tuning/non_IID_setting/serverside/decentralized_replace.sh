#! /bin/sh

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1.\2/')

cp fixed_files/decentralized/traininglog.py poseidon_dec/lib/python$ver/site-packages/kerasplotlib/
