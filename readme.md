# Getting Started

## Linux
This is set up with scripts to run on Linus to install `docker`
and `nvidia-docker` to allow GPU support.

`sudo ./linux/install-docker`

and if you want GPU follow up with

`sudo ./linux/install-nvidia`

If all is well, you will see a listing of your video cards:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 415.25       Driver Version: 415.25       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN RTX           Off  | 00000000:03:00.0  On |                  N/A |
| 41%   37C    P8     9W / 280W |   1036MiB / 24165MiB |     13%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## Mac

Run this to use `brew` to get docker installed.

`./mac/install-docker`


# Running Notebooks

This is configured to run with `docker-compose`, so just start things up with

`docker-compose up`

And then you can just go to http://localhost:8888 to get started. All the notbook security is 
switched off as this is packed up for learning purposes, one less thing to worry about.
