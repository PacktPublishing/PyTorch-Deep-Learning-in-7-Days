# PyTorch Deep Learning in 7 Days[Video]
This is the code repository for [PyTorch Deep Learning in 7 Days[Video]](https://www.packtpub.com), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the video course from start to finish.
## About the Video Course
PyTorch is Facebook’s latest Python-based framework for Deep Learning. It has the ability to create dynamic Neural Networks on CPUs and GPUs, both with a significantly less code compared to other competing frameworks. PyTorch has a unique interface that makes it as easy to learn as NumPy.

This 7-day course is for those who are in a hurry to get started with PyTorch. You will be introduced to the most commonly used Deep Learning models, techniques, and algorithms through PyTorch code. This course is an attempt to break the myth that Deep Learning is complicated and show you that with the right choice of tools combined with a simple and intuitive explanation of core concepts, Deep Learning is as accessible as any other application development technologies out there. It’s a journey from diving deep into the fundamentals to getting acquainted with the advance concepts such as Transfer Learning, Natural Language Processing and implementation of Generative Adversarial Networks. 

By the end of the course, you will be able to build Deep Learning applications with PyTorch.

<H2>What You Will Learn</H2>
<DIV class=book-info-will-learn-text>
<UL>
<LI>Get comfortable with most commonly used PyTorch concepts, modules and API including Tensor operations, data representations, and manipulation
<LI>Work with Deep Learning models and architectures including layers, activations, loss functions, gradients, chain rule, forward and backward passes, and optimizers
<LI>Apply Deep Learning architectures to solve Machine Learning problems for Structured Datasets, Computer Vision, and Natural Language Processing
<LI>Utilize the concept of Transfer Learning by using pre-trained Deep Learning models to your own problems
<LI>Implement state of the art in Natural Language Processing to solve real-world problems such as sentiment analysis
<LI>Implement a simple Generative Adversarial Network to generate fancy images after training on a large image dataset  </LI></UL></DIV>

## Instructions and Navigation
### Assumed Knowledge
This course is for software development professionals and machine learning enthusiasts, who have heard the hype of Deep Learning and want to learn it to stay relevant in their field. Basic knowledge of machine learning concepts and Python programming is required.

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


### Technical Requirements
This course has the following software requirements:<br/>
OS: Any compatible flavor of Linux from the link in minimum requirements<br/>

Browser: Google Chrome, Firefox latest version<br/>

Others: Pytorch (Pytorch.org), Anaconda distribution for Python 3.6 and above from https://repo.continuum.io/archive<br/>

## Related Products
* [Deep Learning with PyTorch [Video]](https://prod.packtpub.com/in/big-data-and-business-intelligence/deep-learning-pytorch-video)

* [Deep Learning Projects with PyTorch [Video]](https://prod.packtpub.com/in/application-development/deep-learning-projects-pytorch-video)

* [Deep Learning with PyTorch Quick Start Guide](https://prod.packtpub.com/in/big-data-and-business-intelligence/deep-learning-pytorch-quick-start-guide)


