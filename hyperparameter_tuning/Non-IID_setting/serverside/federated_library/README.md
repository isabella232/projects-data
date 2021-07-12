# Non-IID library

## dataset_loader.py
###Description:

Load a dataset and prepare training split, test split and dataset info.

###Function:
```load_tf_dataset(dataset_name, decentralized, skew_type, display)```
####Example:
From notebooks: ```ds, (x_test, y_test), ds_info = load_tf_dataset(dataset_name="mnist", decentralized=False, skew_type="qty", display=True)```

## display_distribution.py

###Description:

Functions that can be used to display label distribution of the dataset, label distribution per client or heatmap of the label distribution amongst the clients.

###Function:
```display_dataset_barplot(labels, num_classes, title="")```
####Example:
From ```dataset_loader.load_tf_dataset, l.43```: ```display_dataset_barplot(y_train, ds_info['num_classes'], "Train split")```
###Function:
```display_per_client_barplot(clientsDataLabels, num_clients, num_classes, decentralized)```
####Example:
From ```distributions.build_ClientData_from_dataidx_map, l.72```: ```display_per_client_barplot(clientsDataLabels, num_classes=num_classes, num_clients=num_clients, decentralized=decentralized)```
###Function:
```display_heatmap(dirimap, labels, num_clients, num_classes, decentralized)```
####Example:
From ```distributions.build_ClientData_from_dataidx_map, l.71```: ```display_heatmap(dataidx_map, num_classes=num_classes, num_clients=num_clients, labels=y_train, decentralized=decentralized)```


## distributions.py

### Description:

Functions to distribute the dataset with a skew (qty, label, feature) or not (iid_distrib).

###Function:
```to_ClientData(clientsData: np.ndarray, clientsDataLabels: np.ndarray, ds_info, train=True)```
####Example:
From ```distributions.build_ClientData_from_dataidx_map, l.75```: ```to_ClientData(clientsData, clientsDataLabels, ds_info)``` ```net_dataidx_map``` is an array of length equals to the number of clients, each entry contains an array holding the samples' IDs assigned to each client.
###Function:
```build_ClientData_from_dataidx_map(x_train: np.ndarray, y_train: np.ndarray, ds_info, dataidx_map, decentralized, display)```
####Example:
From ```distributions.qty_skew_distrib, l.151```: ```build_ClientData_from_dataidx_map(x_train, y_train, ds_info, net_dataidx_map, decentralized=decentralized, display=display)```. ```net_dataidx_map``` is an array of length equals to the number of clients, each entry contains an array holding the samples' IDs assigned to each client.
###Function:
```iid_distrib(x_train: np.ndarray, y_train: np.ndarray, ds_info, decentralized, display=False)```
####Example:
From ```fedAvg_training.train_fedAvg, l.35```: ```iid_distrib(x_train, y_train, ds_info, decentralized=False, display=display)```
###Function:
```qty_skew_distrib(x_train: np.ndarray, y_train: np.ndarray, ds_info, beta, decentralized, display=False)```
####Example:
From ```fedAvg_training.train_fedAvg, l.26```: ```qty_skew_distrib(x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)```.```params["skew"]``` is the parameter for the Dirichlet distribution (```float``` or ```int```).
###Function:
```label_skew_distrib(x_train: np.ndarray, y_train: np.ndarray, ds_info, beta, decentralized, display=False)```
####Example:
From ```fedAvg_training.train_fedAvg, l.29```: ```label_skew_distrib(x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)```. ```params["skew"]``` is the parameter for the Dirichlet distribution (```float``` or ```int```).
###Function:
```feature_skew_distrib(x_train: np.ndarray, y_train: np.ndarray, ds_info, sigma, decentralized, display=False)```
####Example:
From ```fedAvg_training.train_fedAvg, l.32```: ```feature_skew_distrib(x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)```.```params["skew"]``` is the parameter for the Gaussian distribution (```float``` or ```int```).

## fedAvg_training.py

###Description:

Run training and test with FedJAX FedAvg.

###Function:
```train_fedAvg(params, ds, test_split, ds_info, custom_model=None, display=False)```
####Example:
From fedJAX notebooks: ```train_fedAvg(params, ds, test_split, ds_info, display=display)```. ```params``` is a ```dict``` containing the following entries ```act_fn, client_lr, server_lr, client_momentum, server_momentum, batch_size, epochs_per_round, rounds, skew```.

## haiku_dropout.py

###Description:

Custom haiku Dropout (hk.Module) class since dropout is implemented in a weird way in haiku.

###Class
```Dropout```
####Example:
From ```models_haiku.get_model, l.71```: ```Dropout(0.2, seed)```

## metrics.py

###Description:

Metrics functions to be used with FedJAX.

###Function:
```accuracy(batch: core.Batch, preds: jnp.ndarray)```
####Example:
From ```models_haiku.get_model, l.31```: ```metrics = collections.OrderedDict(accuracy=accuracy)```

###Function:
```mse(batch: core.Batch, preds: jnp.ndarray)```
####Example:
From ```models_haiku.get_model, l.103-107```: ```core.create_model_from_haiku(..., loss_fn=mse, metrics_fn_map=metrics)```

##models_haiku.py

###Description:

Get default haiku model for mnist, emnist, svhn or cifar10 or a custom haiku model.

###Function:
```get_model(params, ds_info, custom_model=None)```
####Example:
From ```fedAvg_training.train_fedAvg, l.37```: ```model = get_model(params, ds_info, custom_model)```. If needed, ```custom_model``` must be a ```hk.Sequential``` model.

##models_keras.py

###Description:

Get default keras model for mnist, emnist, svhn or cifar10 or a custom keras model.

###Function:
```get_model(params, ds_info, custom_model=None)```
####Example:
From decentralized notebook, ```experiment``` function: ```model = get_model(params, ds_info)```. If needed, ```custom_model``` must be a tensorflow/keras ```Sequential``` model.