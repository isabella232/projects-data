#! /bin/sh

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1.\2/')

cp fixed_files/fedjax/typing.py poseidon_fed/lib/python$ver/site-packages/fedjax/core/
cp fixed_files/fedjax/client_trainer.py poseidon_fed/lib/python$ver/site-packages/fedjax/core/
