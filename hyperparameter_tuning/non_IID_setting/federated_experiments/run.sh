# !/bin/bash

# Launch jupyter
nohup jupyter notebook --no-browser --allow-root --port=8891 &
echo "Jupyter notebook launched on port 8891"

sleep 2

# Get notebook token
jupyter notebook list