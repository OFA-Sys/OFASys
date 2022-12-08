<p align="center">
  <img src="https://avatars.githubusercontent.com/u/98636793?s=200&v=4" width="150">
  <br />
  <a href="https://ofasys.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/ofasys/badge/?version=latest"/></a>
</p>

# What is OFASys?

<img src="images/task7.gif" width = "700" alt="" align=center />

OFASys is a multi-modal multi-task learning system designed to make multi-modal tasks declarative, modular and task-scalable. With OFASys, it is easy to:

- Rapidly introduce new multi-modal tasks/datasets by defining a declarative one-line instruction.
- Develop new or reuse existing modality-specific components.
- Jointly train multiple multi-modal tasks together without manual processing of multi-modal data collating.


# Requirements

- PyTorch version >= 1.8.0
- Python version >= 3.6
- Torchaudio >= 0.8.0

# Installation

## Install with pip

Through the pip installation, users can experience the basic multi-task training and inference functions of OFASys.

```
pip install http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/pkg/ofasys-0.1.0-py3-none-any.whl
```

Test your installation.

```
python -c "import ofasys"
```

Using the audio feature in OFASys requires the [soundfile](https://github.com/bastibe/python-soundfile#installation) library to be installed.
In the Ubuntu OS, run the following command:

```
sudo apt-get update
sudo apt-get install libsndfile1
```

## Install with source (Optional)

Users can install OFASys from the source code to customize their training tasks and full functions.

```
git clone https://github.com/OFA-Sys/OFASys.git
cd OFASys
python setup.py develop
```

# Getting Started

The [documents](https://ofasys.readthedocs.io/en/latest/start/quickstart.html) contains more instructions for getting started.
