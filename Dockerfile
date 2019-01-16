FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
/bin/bash ~/anaconda.sh -b -p /opt/conda && \
rm ~/anaconda.sh && \
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
conda install pytorch torchvision cuda100 -c pytorch && \
echo "conda activate base" >> ~/.bashrc

#all the code samples for the video series
VOLUME ["/src"]

#serve up a jupyter notebook 
WORKDIR /src
EXPOSE 8888

#this has security disabled which is less fuss for learning purposes
CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True
