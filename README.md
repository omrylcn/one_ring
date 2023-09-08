# Tensorflow Segmentation


<img title="ring" alt="ring" src="images/ring.png" height=255> <img title="tf_seg" alt="tf_seg" src="images/tf_seg.png" height=245>




```tf_seg``` is a light, moduler,configurable segmentation wrapper package for training,deployment and inference part of ML Life Cycle. Also it can be a template module for any Computer Vision project to use production. It is designed to be modular and reproducible. It is a good choice for experiment tracking and development.

## How to Build
In order to start building your segmentation projects with tf_seg you need first to build it in your environment. Following these steps you can build easily. Don't worry, it will take only a few minutes before you start delving into details of the module.

``` bash
git clone https://github.com/omrylcn/tf_seg.git
cd tf_seg
conda create -n tf_seg python=3.10
conda activate tf_seg
pip3 install -r requirement.txt
```

Now that we installed all the necessary libraries, we can start by importing the submodules of tf_seg necessary to build our first project as a fresh start.

``` python
from tf_seg.config import get_config
from tf_seg.data import get_data_loader
from tf_seg.model import get_model_builder
```

