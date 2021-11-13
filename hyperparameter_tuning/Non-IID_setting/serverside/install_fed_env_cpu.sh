#! /bin/bash

# Check for python
if ! hash python3; then
    echo "python3 is not installed"
    exit 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "36" ]; then
    echo "This script requires python3 to be at least Python 3.6"
    exit 1
fi

# Check that nohup is installed, install it if not
dpkg -s coreutils &> /dev/null
if [ $? -ne 0 ]; then
    echo "nohup is not installed but is required, the script will install it"
    sudo apt install coreutils
fi

# Clean old environment
echo "Cleaning old environment"
./clean.sh
sleep 1

# Create and activate new environment
echo "Creating fresh virtual environment"
python3 -m venv poseidon_fed
source poseidon_fed/bin/activate

# Install required packages
echo "Installing required packages"

pip install --upgrade pip &> /dev/null
echo -ne "1/11 \r"
pip install --upgrade wheel &> /dev/null
echo -ne "2/11 \r"
pip install --upgrade jupyterlab &> /dev/null
echo -ne "3/11 \r"
pip install --upgrade ipywidgets &> /dev/null
echo -ne "4/11 \r"
pip install --upgrade numpy pandas matplotlib &> /dev/null
echo -ne "7/11 \r"
pip install --upgrade tensorflow_datasets &> /dev/null
echo -ne "8/11 \r"
pip install --upgrade jax jaxlib &> /dev/null
echo -ne "10/11 \r"
pip install fedjax==0.0.5 &> /dev/null
echo -ne "11/11 \r"
echo -ne "\n"

# Replace files
./fedjax_replace.sh


# Launch jupyter
./run.sh
