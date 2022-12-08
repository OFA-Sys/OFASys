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

## Training One Model for All Tasks

###Define the tasks

OFASys can co-train multiple multi-modal tasks flexibly.


    >>> from ofasys import Task, Trainer, GeneralistModel
    ... task1 = Task(
    ...     name='caption',
    ...     instruction='[IMAGE:image_url] what does the image describe? -> [TEXT:caption]',
    ...     micro_batch_size=4,
    ... )
    >>> task2 = Task(
    ...     name='text_infilling',
    ...     instruction='what is the complete text of " [TEXT:sentence,mask_ratio=0.3] "? -> [TEXT:sentence]',
    ...     micro_batch_size=2,
    ... )

In the simplest scenario, you only need to specify an instruction to define your task and a task name as an identifier.
For more details about how to define a task for training, see [Define a Task](https://ofasys.readthedocs.io/en/latest/howto/add_task.html) and [Train a task](https://ofasys.readthedocs.io/en/latest/howto/train.html).

### Set the Dataset

The Task can use a regular Pytorch Dataloader which can be constructed by Huggingface Dataset or a customized Pytorch Dataset.

    >>> from datasets import load_dataset
    >>> task1.add_dataset(load_dataset('TheFusion21/PokemonCards')['train'], 'train')
    >>> task2.add_dataset(load_dataset('glue', 'cola')['train'], 'train')
    
### Create a Generalist Model and Train all Tasks Together

The GeneralistModel of OFASys (OFA+) is capable of handling multiple [modalities](https://ofasys.readthedocs.io/en/latest/concept/plan.html#modality) including:
*TEXT*, *IMAGE*, *AUDIO*, *VIDEO*, *MOTION*, *BOX*, *PHONE*.

The OFASys Trainer “mixes” multiple Tasks with any dataset and abstracts away all the engineering complexity needed for scale.


    >>> model = GeneralistModel()
    >>> trainer = Trainer()
    >>> trainer.fit(model=model, tasks=[task1, task2])

The complete script is available at [scripts/trainer_api.py](https://github.com/OFA-Sys/OFASys/blob/main/scripts/trainer_api.py).

## Infer multiple multi-modal tasks with One Checkpoint

OFASys can infer multiple multi-modal tasks using just **One** checkpoint.

    >>> from ofasys import OFASys
    >>> model = OFASys.from_pretrained('multitask.pt')

OFASys enables multi-task multi-modal inference through the instruction alone. Let's go through a couple of examples!
    
### Image Captioning

<img src="https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/val2014/COCO_val2014_000000222628.jpg" width="400">

    >>> instruction = '[IMAGE:img] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    >>> data = {'img': "./COCO_val2014_000000222628.jpg"}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)
        # "a man and woman sitting in front of a laptop computer"


### Visual Grounding

<img src="https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg" width="400">

    >>> instruction = '[IMAGE:img] <BOS> which region does the text " [TEXT:cap] " describe? <EOS> -> [BOX:patch_boxes,add_bos,add_eos]'
    >>> data = {'img': "https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg", "cap": "hand"}
    >>> output = model.inference(instruction, data=data)
    >>> output.save_box("output.jpg")

<img src="http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/inference_caption_0.jpg" width="400">

### Text Summarization

    >>> instruction = '<BOS> what is the summary of article " [TEXT:src] "? <EOS> -> <BOS> [TEXT:tgt] <EOS>'
    >>> data = {'src': "poland 's main opposition party tuesday endorsed president lech walesa in an upcoming "
    ...        "presidential run-off election after a reformed communist won the first round of voting ."}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)
        # "polish opposition endorses walesa in presidential run-off"

### Table-to-Text Generation

    >>> instruction = '<BOS> structured knowledge: " [STRUCT:database,uncased] "  . how to describe the tripleset ? <EOS> -> <BOS> [TEXT:tgt] <EOS>'
    >>> data = {
    ...     'database': [['Atlanta', 'OFFICIAL_POPULATION', '5,457,831'],
    ...                  ['[TABLECONTEXT]', 'METROPOLITAN_AREA', 'Atlanta'],
    ...                  ['5,457,831', 'YEAR', '2012'],
    ...                  ['[TABLECONTEXT]', '[TITLE]', 'List of metropolitan areas by population'],
    ...                  ['Atlanta', 'COUNTRY', 'United States'],
    ...     ]
    ... }
    >>> output = model.inference(instruction, data=data, beam_size=1)
    >>> print(output.text)
        # "atlanta is the metropolitan area in the united states in 2012."

### Text-to-SQL Generation

    >>> instruction = '<BOS> " [TEXT:src] " ; structured knowledge: " [STRUCT:database,max_length=876] " . generating sql code. <EOS> -> <BOS> [TEXT:tgt] <EOS>'
    >>> database = [
    ...             ['concert_singer'],
    ...             ['stadium', 'stadium_id , location , name , capacity , highest , lowest , average'],
    ...             ['singer', 'singer_id , name , country , song_name , song_release_year , age , is_male'],
    ...             ['concert', 'concert_id , concert_name , theme , stadium_id , year'],
    ...             ['singer_in_concert', 'concert_id , singer_id']
    ... ]
    >>> data = [
    ...     {'src': 'What are the names, countries, and ages for every singer in descending order of age?', 'database': database},
    ...     {'src': 'What are all distinct countries where singers above age 20 are from?', 'database': database},
    ...     {'src': 'What are the locations and names of all stations with capacity between 5000 and 10000?', 'database': database}
    ... ]
    >>> output = model.inference(instruction, data=data)
    >>> print('\n'.join(o.text for o in output))]
        # "select name, country, age from singer order by age desc"
        # "select distinct country from singer where age > 20"
        # "select location, name from stadium where capacity between 5000 and 10000"
    
### Video Captioning
   
<img src="https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/video7021.gif" width="400">

    >>> instruction = '[VIDEO:video] <BOS> what does the video describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    >>> data = {'video': './video7021.mp4'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)
        # "a baseball player is hitting a ball"
        
### Speech-to-Text Generation

<audio controls="controls">
  <source src="http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac" type="audio/wav">
  Your browser does not support the <code>audio</code> element.
</audio>
    
    >>> instruction = '[AUDIO:wav] <BOS> what is the text corresponding to the voice? <EOS> -> [TEXT:text,preprocess=text_phone,add_bos,add_eos]'
    >>> data = {'wav': './1272-128104-0001.flac'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)
        # "nor is mister klohs manner less interesting than his manner"
        
### Text-to-Image Generation

    >>> instruction = 'what is the complete image? caption: [TEXT:text]"? -> [IMAGE,preprocess=image_vqgan,adaptor=image_vqgan]'
    >>> data = {'text': "a city with tall buildings and a large green park."}
    >>> output = model.inference(instruction, data=data)
    >>> output[0].save_image('0.png')

<img src="https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/image-gen_example.png" width="400">
