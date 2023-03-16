ip addr add 192.168.140.3/24 dev ens37  
ip route add default via 192.168.137.254 dev eno1


#### gcc 

sudo apt install build-essential -y
sudo apt install software-properties-common -y 
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9 gcc-11 g++-11 -y

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo update-alternatives --config gcc

##############
pip install -r ./requirement.txt    -i https://pypi.tuna.tsinghua.edu.cn/simple




## vnc

sudo apt install tasksel -y
apt install tigervnc-standalone-server tigervnc-common -y 
sudo apt-get install gnome-panel -y 


##  base 
apt-get  update -y 
apt-get install   vim  openssh-server   ipmitool   htop   gcc   g++    make   sysstat    net-tools   curl   wget   screen   -y
vim /etc/ssh/sshd_config 

## docker and nvidia-docker

apt  install docker.io -y  
systemctl enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update -y 
sudo apt-get install -y nvidia-docker2



## nvidia-docker离线安装依赖的几个包
libnvidia-container-tools
libnvidia-container1  
nvidia-container-runtime_3.5.0-1_amd64.deb
nvidia-container-toolkit_1.5.1-1_amd64.deb
nvidia-docker2_2.6.0-1_all.deb




## network 
vim  /etc/netplan/01-network-manager-all.yaml 

network:
  version: 2
  renderer: networkd
  ethernets:
    ens33:   
      dhcp4: no    
      dhcp6: yes
      addresses: [192.168.1.55/24]   
      gateway4: 192.168.1.254 
      nameservers:
          addresses: [114.114.114.114, 8.8.8.8] 

## driver
ubuntu-drivers devices  ## 检查显卡型号
lshw -numeric -C display  ## 检查显卡型号
sudo /etc/init.d/gdm stop
sudo /etc/init.d/gdm status
sudo /etc/init.d/lightdm stop
sudo /etc/init.d/lightdm status
sudo service lightdm stop

$ cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
blacklist nouveau
options nouveau modeset=0

sudo update-initramfs -u
lsmod | grep nouveau 

sh    --no-opengl-files


## anaconda3

vim ~/.condarc


channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud


export PATH="/home/name/anaconda3/bin:$PATH"

conda clean -i 
conda info


wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run


# set-save-dir 
# vim  .condarc
########################
envs_dirs:
  - /opt/anaconda3/envs
pkgs_dirs:
  - /opt/anaconda3/pkgs 
##########################

#### init conda 


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


## cuda 11.2
vim  .bashrc
export PATH="/usr/local/cuda-10.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"

nvcc -V



## cdunn 
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.2/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64
sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h /usr/local/cuda-10.2/lib64/libcudnn*


sudo cp cuda/include/cudnn.h /usr/local/cuda-11.2/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64
sudo chmod a+r /usr/local/cuda-11.2/include/cudnn.h /usr/local/cuda-11.2/lib64/libcudnn*

## pytorch

conda create -n pytorch_test01 python=3.7
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

++++++++++++++++++++++++++++++++++++++++++++++

import torch
torch.cuda.is_available()
torch.zeros(1).cuda()

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda()) 

++++++++++++++++++++++++++++++++++++++++++++++




## tensorflow
conda create -n tensorflow python=3.7
conda activate tensorflow 
conda install  tensorflow-gpu 
conda install  tensorflow==2.4.1
conda  list

++++++++++++++++++++++++++++++++++++++++++++++++

import tensorflow as tf
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

++++++++++++++++++++++++++++++++++++++++++++++++

## caffe
conda create -n caffe python=3.6
conda activate caffe
conda install -c defaults caffe-gpu






## docker

curl -sSL https://get.daocloud.io/docker | sh

curl -L https://get.daocloud.io/docker/compose/releases/download/1.29.2/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose




## docker-fast

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://dwa4f2kp.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker



## docker-DCGM


distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update -y
sudo apt-get install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart docker

sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
docker run -d --gpus all --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:2.2.9-2.5.0-ubuntu20.04



