#! /bin/bash

echo "Killing notebooks"
pkill -f "jupyter.*8890"
pkill -f "jupyter.*8891"

echo "Remove environments and files"
rm -rf ~/poseidon_dec/
rm -rf ~/poseidon_fed/
rm -rf __pycache__/
rm -rf nohup.out
