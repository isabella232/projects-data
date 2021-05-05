# IID setting scripts

## Requirements
```
tensorflow
tensorflow_datasets
pandas
talos
```


## fedAvg_gridSearch_IID

This notebook finds the best hyperparameters for each client and averages them. Then it tries ```relu,tanh``` and ```sigmoid``` activation functions with the best averaged hyperparameters and outputs which one is the best fit.


Specify the number of clients, number of classes, dataset name in [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview) and experiment name.

Specify the gridsearch parameters in ```get_params_for_X_clients()``` and the model in ```get_model()```.

Run the notebook.

The notebook will output the averaged best parameters, the best test accuracy with those parameters amongst the clients and the best activation function.

Complete results of the gridsearch of hyperparameters will be in ```experiment_name+"_res/res"+str(numClients))/``` folder.

## fedAvg_intervalSearch_IID

This notebook follows ```fedAvg_gridSearch_IID``` notebook. Once we've found the best averaged hyperparameters and activation function, it will find the best interval for the approximated activation function.

Specify the approximated activation function in ```approx_act_fn```, intervals for the approximated activation function in ```intervals```, the hyperparameters in ```intervals_params``` and the model in ```get_model()```.

Run the notebook.

The notebook will output the proportion of each interval that were best for each client as well as the best interval amongst the client, and the best test accuracy amongst the clients with this interval.

Complete results of the gridsearch of intervals will be in ```experiment_name+"_res/intervals_res"+str(numClients))/``` folder.

## ADAMApprox_IID

This notebook follow ```fedAvg_intervalSearch_IID``` notebook.

Specify the approximated activation function in ```act_fn```, the interval for the approximated activation function in ```interval```, the hyperparameters in ```get_params_for_X_clients()``` and the model in ```get_model()```.

Specify the precisions you want to test in ```precisions``` (0 for 0 digit after the point, 2 for 2 digits after the point, ..., full for full precision). Here we run the experiment 10 times for each precision to get a meaningful average value of loss, accuracy, presicion and recall.

Run the notebook.

```coeff_range``` is the range the coeff ```1 / (sqrt(v_hat) + epsilon)``` takes under full precision.

Final result is the averaged loss, accuracy, presicion and recall for each precision value.