============
Installation
============

Requirements
------------

- PyTorch version >= 1.8.0
- Python version >= 3.6
- Torchaudio >= 0.8.0

First, we need a ``python`` environment whose version is at least greater than ``3.6.0``.
If you don't have one, please refer to the `documentation <https://docs.anaconda.com/anaconda/install/>`_ to install and configure the Anaconda environment.

.. code-block::

   conda create -n ofasys python=3.8
   conda activate ofasys

Then, install `Pytorch <https://pytorch.org/get-started/locally/>`_ and keep the version at least greater than ``1.8.0``.

.. code-block::

   pip install torch torchvision torchaudio


.. _installation:

Installation
------------

1. Install with pip
~~~~~~~~~~~~~~~~~~~

Through the pip installation, users can experience the basic multi-task training and inference :ref:`functions<anchor_quickstart>` of OFASys.

.. code-block::

    pip install -U http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/pkg/ofasys-0.1.0-py3-none-any.whl

Test the installation.

.. code-block::

    python -c "import ofasys"

Using the audio feature in OFASys requires the `soundfile <https://github.com/bastibe/python-soundfile#installation>`_ library to be installed.
In the Ubuntu OS, run the following command:

.. code-block::

    sudo apt-get update
    sudo apt-get install libsndfile1

2. Install with source (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install OFASys from the source code to customize their training tasks and full functions.

.. code-block::

    git clone https://github.com/OFA-Sys/OFASys.git
    cd OFASys
    python setup.py develop


