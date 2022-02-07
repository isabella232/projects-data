# UBUNTU 20.04

 `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`

## Install CUDA

[Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-nvidia-driver-and-cuda-software)

```{bash}
sudo apt update
sudo apt upgrade
sudo apt -y install gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub
sudo apt update
sudo apt -y install cuda

nano ~/.bashrc
export PATH="/usr/local/cuda-11.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH"
```

## Instal cudNN

First, download [file](https://drive.google.com/file/d/1gVovwfd58lmZS1VQSjNxWmkpxqJX8GMD/view?usp=sharing)

```{bash}
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub
sudo apt update
sudo apt -y install libcudnn8=8.3.1.22-1+cuda11.5

sudo reboot
```

## Set up repository

```{bash}
sudo apt -y install git
sudo apt -y install python3-venv

git clone https://github.com/ldsec/projects-data.git

sudo apt -y install python3-venv
./projects-data/hyperparameter_tuning/non_IID_setting/federated_experiments/install_fed_env.sh gpu
source ~/poseidon_fed/bin/activate
```

## Run python scripts in parallel

Mask other GPUs

`export CUDA_VISIBLE_DEVICES=x`,
with x number between 0 and 4

## Install Python with specific version

```{bash}
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-dev python3.8-venv python-wheel
```

## Uninstall CUDA and cudNN

```{bash}
sudo apt -y remove cuda
sudo apt -y remove libcudnn8=8.3.1.22-1+cuda11.5
sudo apt -y remove nvidia-driver-495
sudo apt purge nvidia-*
sudo apt autoremove
rm -r /usr/local/cuda-11.5/
rm -r /usr/local/cuda-11/
rm -r /usr/local/cuda/
```
