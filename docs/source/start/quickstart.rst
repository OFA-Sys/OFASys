===================
Usage in 15 minutes
===================

.. _anchor_quickstart:

Training One Model for All Tasks
================================

1. Define the tasks
-------------------

OFASys can co-train multiple multi-modal tasks flexibly.

.. code:: python

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
For more details about how to define a task for training, see :doc:`../howto/add_task` and :doc:`../howto/train`.

2. Set the Dataset
-------------------

The Task can use a regular Pytorch Dataloader which can be constructed by Huggingface Dataset or a customized Pytorch Dataset.

.. code:: python

    >>> from datasets import load_dataset
    >>> task1.add_dataset(load_dataset('TheFusion21/PokemonCards')['train'], 'train')
    >>> task2.add_dataset(load_dataset('glue', 'cola')['train'], 'train')

3. Create a Generalist Model and Train all Tasks Together
----------------------------------------------------------

The GeneralistModel of OFASys (OFA+) is capable of handling multiple :ref:`modalities<anchor_modalities>` including:
*TEXT*, *IMAGE*, *AUDIO*, *VIDEO*, *MOTION*, *BOX*, *PHONE*.

The OFASys Trainer “mixes” multiple Tasks with any dataset and abstracts away all the engineering complexity needed for scale.

.. code:: python

    >>> model = GeneralistModel()
    >>> trainer = Trainer()
    >>> trainer.fit(model=model, tasks=[task1, task2])

.. _anchor_train_the_model:

The complete script is available at `scripts/trainer_api.py <https://github.com/OFA-Sys/OFASys/blob/main/scripts/trainer_api.py>`_.
More details on how to write YAML files to define tasks and more distributed usage can be found in :doc:`../howto/train`.

Inference with All Kinds of Tasks with One Checkpoint
======================================================

OFASys can infer multiple multi-modal tasks using just **One** checkpoint.

Load a multi-task checkpoint

.. code:: python

    >>> from ofasys import OFASys
    >>> model = OFASys.from_pretrained('http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/model_hub/multitask_10k.pt')
    >>> model = model.cuda()  # Omit this line if you don't have a GPU

OFASys enables multi-task multi-modal inference through the instruction alone. Let's go through a couple of examples!

Image Captioning
----------------

.. code:: python

    >>> instruction = '[IMAGE:img] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    >>> data = {'img': "https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/val2014/COCO_val2014_000000222628.jpg"}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/val2014/COCO_val2014_000000222628.jpg

::

   "a man and woman sitting in front of a laptop computer"

Visual Grounding
----------------

.. code:: python

    >>> instruction = '[IMAGE:img] <BOS> which region does the text " [TEXT:cap] " describe? <EOS> -> [BOX:patch_boxes,add_bos,add_eos]'
    >>> data = [
    ...     {'img': "https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg", 'cap': 'hand'},
    ...     {'img': "http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/train2014/COCO_train2014_000000581563.jpg", 'cap': 'taxi'},
    ... ]
    >>> output = model.inference(instruction, data=data)
    >>> for i, out in enumerate(output):
    ...     out.save_box(f'{i}.jpg')

.. image:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/inference_caption_0.jpg
.. image:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/inference_caption_1.jpg

Text Summarization
-------------------

.. code:: python

    >>> instruction = '<BOS> what is the summary of article " [TEXT:src] "? <EOS> -> <BOS> [TEXT:tgt] <EOS>'
    >>> data = {'src': "poland 's main opposition party tuesday endorsed president lech walesa in an upcoming "
    ...        "presidential run-off election after a reformed communist won the first round of voting ."}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

::

   "polish opposition endorses walesa in presidential run-off"

Table-to-Text Generation
--------------------------
+---------------+--------------------+-----------------------------------------+
| Atlanta       | OFFICIAL_POPULATION| 5,457,831                               |
+---------------+--------------------+-----------------------------------------+
|[TABLECONTEXT] | METROPOLITAN_AREA  |  Atlanta                                |
+---------------+--------------------+-----------------------------------------+
| 5,457,831     | YEAR               | 2012                                    |
+---------------+--------------------+-----------------------------------------+
| [TABLECONTEXT]|  [TITLE]           | List of metropolitan areas by population|
+---------------+--------------------+-----------------------------------------+
|Atlanta        | COUNTRY            | United States                           |
+---------------+--------------------+-----------------------------------------+

.. code:: python

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

::

   "atlanta is the metropolitan area in the united states in 2012."

Text-to-SQL Generation
---------------------------
+----------------------------------------------------------------------------------------+
|                 Database: concert_singer                                               |
+------------------+---------------------------------------------------------------------+
| Table            | Fields                                                              |
+==================+=====================================================================+
|stadium           | stadium_id, location, name, capacity, highest, lowest, average      |
+------------------+---------------------------------------------------------------------+
| singer           | singer_id, name, country, song_name, song_release_year, age, is_male|
+------------------+---------------------------------------------------------------------+
| concert          | concert_id, concert_name, theme, stadium_id, year                   |
+------------------+---------------------------------------------------------------------+
|singer_in_concert | concert_id, singer_id                                               |
+------------------+---------------------------------------------------------------------+

.. code:: python

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
    >>> print('\n'.join(o.text for o in output))

::

    "select name, country, age from singer order by age desc"
    "select distinct country from singer where age > 20"
    "select location, name from stadium where capacity between 5000 and 10000"

Video Captioning
------------------

.. code:: python

    >>> instruction = '[VIDEO:video] <BOS> what does the video describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    >>> data = {'video': 'oss://ofasys/datasets/msrvtt_data/videos/video7021.mp4'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

.. video:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/datasets/msrvtt_data/videos/video7021.mp4

::

   "a baseball player is hitting a ball"

Speech-to-Text Generation
----------------------------

.. code:: python

    >>> instruction = '[AUDIO:wav] <BOS> what is the text corresponding to the voice? <EOS> -> [TEXT:text,preprocess=text_phone,add_bos,add_eos]'
    >>> data = {'wav': 'oss://ofasys/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

.. raw:: html

    <audio controls="controls">
      <source src="http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac" type="audio/wav">
      Your browser does not support the <code>audio</code> element.
    </audio>


::

   "nor is mister klohs manner less interesting than his manner"

Text-to-Image Generation
-------------------------
.. code:: python

    >>> instruction = 'what is the complete image? caption: [TEXT:text]"? -> [IMAGE,preprocess=image_vqgan,adaptor=image_vqgan]'
    >>> data = {'text': "a city with tall buildings and a large green park."}
    >>> output = model.inference(instruction, data=data)
    >>> output[0].save_image('0.png')

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/image-gen_example.png


The complete script is available at `scripts/inference_multiple_task.py <https://github.com/OFA-Sys/OFASys/blob/main/scripts/inference_multiple_task.py>`_


