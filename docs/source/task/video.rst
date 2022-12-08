Video-Related Tasks
=====================

.. _videoclass:

Video Classification
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
The video classification task is a fundamental task in the field of video understanding where the model needs to predict the label for a given video clip.
We evaluate our model on the Kinetics-400 dataset, which contains ~300k video clips from 400 classes.
We report the accuracy on the val split of the Kinetics-400 dataset.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[VIDEO:video] <BOS> what is in the video?  <EOS> -> <BOS> [TEXT:label_name,closed_set] <EOS>

Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '[VIDEO:video] <BOS> what is in the video?  <EOS> -> <BOS> [TEXT:label_name,closed_set] <EOS>'
    >>> data = {'video': 'oss://ofasys/datasets/kinetics_data/k400.256p/test/-7aeB7vFtB4_000037_000047.mp4'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)


CASE
^^^^^^^^^^^^^^^^^^
input:

.. video:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/datasets/kinetics_data/k400.256p/test/-7aeB7vFtB4_000037_000047.mp4

output:

::

   "playing the piano"

.. _videocaption:

Video Captioning
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
The video captioning task requires the model to generate a textual description for a given video clip.
We evaluate the proposed method on MSR-VTT caption dataset, which contains 10K video clips 200K descriptions of the videos.
Following, We report CIDEr scores on the val split of the MSR-VTT dataset.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[VIDEO:video] <BOS> what does the video describe? <EOS> -> <BOS> [TEXT:cap] <EOS>

Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '[VIDEO:video] <BOS> what does the video describe? <EOS> -> <BOS> [TEXT:cap]'
    >>> data = {'video': 'oss://ofasys/datasets/msrvtt_data/videos/video7030.mp4'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)


CASE
^^^^^^^^^^^^^^^^^^
input:

.. video:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/datasets/msrvtt_data/videos/video7030.mp4

output:

::

   "a group of people are dancing"


.. _videoqa:

Video Question Answering
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
The video captioning task requires the model to generate a answer for a given video clip and a question related to that video clip.
We evaluate the proposed method on MSR-VTT QA dataset, which contains question-answer pairs extracted from the original MSR-VTT dataset.
We report the accuracy on the val split of MSR-VTT QA dataset.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    [VIDEO:video] <BOS> [TEXT:question] <EOS> -> <BOS> [TEXT:answer,is_label] <EOS>
  
Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '[VIDEO:video] <BOS> [TEXT:question] <EOS> -> <BOS> [TEXT:answer,is_label] <EOS>'
    >>> data = {'video': 'oss://ofasys/datasets/msrvtt_data/videos/video9585.mp4', 'question': 'what is a person decorating?'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)


CASE
^^^^^^^^^^^^^^^^^^
input:

.. video:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/datasets/msrvtt_data/videos/video9585.mp4

::

    'what is a person decorating?'

output:

::

   "a person is decorating a cake"


