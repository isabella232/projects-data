#Non-IID Scripts

# Serverside
## Python environments

Both experiments cannot use the same environment, but can be run either on CPU or GPU.

Make sure python ```venv``` is installed: ```sudo apt install python3-venv```

If you want to kill every running jupyter notebooks: ```pkill jupyter```

### Decentralized

Check Python version with ```python3 -V```. It must be version 3.6.X or 3.7.X. If not, see [here](https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu).

Run the script to install the decentralized virtual environment on the server: ```./install_dec_env.sh```. It will install the environment and launch ```jupyter notebook``` in background on port 8890.

To connect with the notebook on your own computer, do a ssh bridge like this: ```ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8890 {server_user}@{server_address}```. Then go to ```localhost:{LOCAL_PORT}``` on your browser.

Requirements installed by the script : ```pip, wheel, jupyterlab, ipywidgets, numpy, talos, pandas, matplotlib, tensorflow_datasets, tensorflow_federated==0.18.0```


### FedAvg
#### CPU support:

Run the script to install the federated virtual environment on the server: ```./install_fed_env_cpu.sh```. It will install the environment and launch ```jupyter notebook``` in background on port 8891.

Requirements installed by the script : ```pip, wheel, jupyterlab, ipywidgets, numpy, pandas, matplotlib, tensorflow_datasets, jax, jaxlib, fedjax==0.0.5```

To connect with the notebook on your own computer, do a ssh bridge like this: ```ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8891 {server_user}@{server_address}```. Then go to ```localhost:{LOCAL_PORT}``` on your browser.

#### GPU support:

The script assumes CUDA11, you can find CUDA version with ```nvcc --version```. For other versions, change the ```jaxlib``` version in the script in accordance with [this document](https://github.com/google/jax/blob/main/README.md#pip-installation).

Run the script to install the federated virtual environment on the server: ```./install_fed_env_gpu.sh```. It will install the environment and launch ```jupyter notebook``` in background on port 8889.

Requirements installed by the script : ```pip, wheel, jupyterlab, ipywidgets, numpy, pandas, matplotlib, tensorflow_datasets, jax, jaxlib==0.1.67+cuda111, fedjax==0.0.5```

To connect with the notebook on your own computer, do a ssh bridge like this: ```ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8891 {server_user}@{server_address}```. Then go to ```localhost:{LOCAL_PORT}``` on your browser.

## Gridsearch Notebooks

Possible ```skew_type```s are ```qty, label, feature``` and ```iid``` by default.

Possible datasets can be found in [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview).

### Decentralized
##### ```decentralized_non-iid``` notebook


what: perform hyperparameters gridsearch and intervals search in decentralized setting

how:

1. Load dataset and generate test split
    * call ```load_tf_dataset```, then transform the ```ds_test``` with ```tf.data.Dataset.from_tensor_slices``` and generate batches with the ```batch(n)``` function.

2. call the run function with the following arguments:
    * hyperparameters: dict of lists whose keys are **act_fn**, **act_fn_approx** (not mandatory if ```with_intervals``` is set to ```False```), **intervals** (not mandatory if ```with_intervals``` is set to ```False```), **client_lr**, **client_momentum**, **batch_size**, **epochs**, **clients_set**, **skews_set**
    * ds: dataset from ```load_tf_dataset```
    * test_dataset: transformed and batched test dataset from 1.
    * ds_info: dataset information
    * with_intervals: True if you want to do the intervals search
    * display: True if you want to see the graphic representation of the distributions

The script will create a folder named ```{dataset_name}_non_iid_res```. It will be used to receive the results of the gridsearch. 3 types of files (or 2 if ```with_intervals``` is set to ```False```) will be generated:

* text file with "distribution" keyword: number of samples and ratio per client for quantity skew, number of samples of each class per client for label skew
* text file with "intervals" keyword: each line corresponds to a client and its best interval
* other text files: result of the grid search, each line is a tuple (validation_accuracy, hyperparameters used), the file is ordered by validation_accuracy.

You can also tune the callbacks (i.e. early stopping) in the ```experiment``` function.


### FedAvg
#####```fedJAX_gridSearch``` notebook

what: perform hyperparameters gridsearch or test accuracy with FedAvg

how:

1. Load dataset and generate test split
    * call ```load_tf_dataset```, then transform the ```ds_test``` with ```fedjax.create_tf_dataset_for_clients(to_ClientData([x_test], [y_test], ds_info, train=False), ['0']).batch(50)```.

2.  You can either
    * call the ```run``` function with the following arguments to run gridsearch:
        * params: dict of lists whose keys are **act_fn**, **client_lr**, **server_lr**, **client_momentum**, **server_momentum**, **batch_size**, **epochs_per_round**, **rounds**, **runs** (int, not list), **clients_set**, **skews_set**
        * ds: dataset from ```load_tf_dataset```
        * test_split: test dataset from 1.
        * ds_info: dataset information
        * display: True if you want to see the graphic representation of the distributions

    * call the ```run_test``` function to train and test with the given sets of  parameters
        * same as above but params must be build in a different way, keys **client_lr**, **client_momentum**, **batch_size**, **clients_set** and **skews_set** must be a list such that entry ```i``` of each list is a set of parameters to test. You can change this behaviour at the beginning of the ```run_test``` function to add or remove tunable parameters.

The script will output the result of the gridsearch performed by the ```run``` function in a text file with the format ```{dataset_name}_{skew_type}_{skew}_{parties}clients.txt```. Each file contains tuples of (test_accuracy, hyperparameters used), ordered by test_accuracy.

#####```fedJAX_intervalSearch``` notebook

what: perform interval search with FedAvg

how:

1. Load dataset and generate test split
    * call ```load_tf_dataset```, then transform the ```ds_test``` with ```fedjax.create_tf_dataset_for_clients(to_ClientData([x_test], [y_test], ds_info, train=False), ['0']).batch(50)```.

2. call the ```run``` function to perform interval search on the given sets of parameters with the following arguments:
    * params: dict of lists whose keys are **act_fn**, **intervals**, **client_lr**, **server_lr**, **client_momentum**, **server_momentum**, **batch_size**, **epochs_per_round**, **rounds**, **runs** (int, not list), **clients_set**, **skews_set**. Keys **client_lr**, **client_momentum**, **batch_size**, **clients_set** and **skews_set** must be a list such that entry ```i``` of each list is a set of parameters. You can change this behaviour at the beginning of the ```run``` function to add or remove tunable parameters.
    * ds: dataset from ```load_tf_dataset```
    * test_split: test dataset from 1.
    * ds_info: dataset information
    * display: True if you want to see the graphic representation of the distributions

# Results aggregation
## Results aggregation Notebooks

Requirements: ```numpy, pandas```

Choose the ```DATASET```, ```SKEW``` and ```PARTIES``` and run the notebook to perform the decentralized results aggregation and get parameters for FedAvg following the method defined in the ```get_res``` function. You can change the aggregation method at will to try to get better or worse results with FedAvg. You can test the accuracy with ```run_test``` in ```fedJAX_gridSearch```